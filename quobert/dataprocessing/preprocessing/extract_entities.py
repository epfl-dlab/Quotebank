import argparse
import re
import string

import pyspark.sql.functions as F
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import BooleanType

from pygtrie import StringTrie  # type: ignore # Sideloaded in the spark-submit

PUNCT = "".join(x for x in string.punctuation if x not in "[]")
TARGET_QUOTE_TOKEN = "[TARGET_QUOTE]"
MASK_TOKEN = "[MASK]"
BLACK_LIST = {"president", "manager"}


def create_trie(names):
    trie = StringTrie(delimiter="/")
    for (name, qid) in names:
        q_name = eval(qid[0])[1]
        name = [x for x in name.split() if x.lower() not in BLACK_LIST]
        for i in range(len(name)):
            trie["/".join(name[i:]).lower()] = (q_name, qid)
    return trie


def update_entity(q_name, qinfo, i, out):
    if q_name in out:
        out[q_name][0].append(i)
    else:
        out[q_name] = ([i], qinfo)


def reduce_entities(entities):
    out = dict()
    for i, value in entities.items():
        if isinstance(value, list):
            for q_name, qinfo in value:
                update_entity(q_name, qinfo, i, out)
        else:
            try:
                update_entity(value[0], value[1], i, out)
            except IndexError:  # this happens when value = '/', which is rare
                print(i, value, "We had an error here")
    return out


def get_partial_match(trie, key):
    match = list(trie[key:])
    return match if len(match) > 1 else match[0]


def fix_special_tokens(tokens):
    out = []
    current = ""
    for token in tokens:
        if token == "[":
            current = "["
        elif token == "]":
            out.append(current + "]")
            current = ""
        elif current != "":
            current += token
        else:
            out.append(token)
    return out


def fix_punct_tokens(tokens):
    out = []
    for token in tokens:
        if token[-1] in PUNCT:
            out += [token[:-1], token[-1]]
        elif token.endswith("'s"):
            out += [token[:-2], "'s"]
        else:
            out.append(token)
    return out


def get_entity(trie, key):
    entites = get_partial_match(trie, key)
    if not type(entites) == tuple:
        raise Exception(f"{entites} is not a tuple")
    return entites


def find_entites(text: str, trie: StringTrie, mask: str = MASK_TOKEN):
    tokens = text.split()
    tokens = fix_punct_tokens(tokens)
    start = 0
    count = 1  # start at 1, 0 is for the "NO_MATCH"
    entities = dict()
    out = []
    for i in range(len(tokens)):
        key = "/".join(tokens[start : i + 1]).lower()
        if trie.has_subtrie(key):  # Not done yet
            if i == len(tokens) - 1:  # Reached the end of the string
                entities[count] = get_entity(trie, key)
                out.append(mask)
        elif trie.has_key(key):  # noqa: W601 # Find a perfect match
            entities[count] = trie[key]
            count += 1
            out.append(mask)
            start = i + 1
        elif start < i:  # Found partial prefix match before this token
            old_key = "/".join(tokens[start:i]).lower()
            entities[count] = get_entity(trie, old_key)
            count += 1
            out.append(mask)
            if trie.has_node(
                tokens[i].lower()
            ):  # Need to verify that the current token isn't in the Trie
                start = i
            else:
                out.append(tokens[i])
                start = i + 1
        else:  # No match
            out.append(tokens[i])
            start = i + 1
    retokenized = "".join(
        [" " + i if not i.startswith("'") and i not in PUNCT else i for i in out]
    ).strip()
    return retokenized, reduce_entities(entities)


def get_targets(entities, target_entity):
    targets = entities.get(target_entity, None)
    if not targets:
        return [0], False
    return targets[0], len(targets[0]) > 1


def check_speaker_in_entities(speaker, names):
    if speaker in (
        "-1",
        "none",
        "not_quote",
        "not_mentioned",
        "not_en",
        "ambiguous",
        "other",
    ):
        return True
    for (name, qid) in names:
        q_name = eval(qid[0])[1]
        if q_name == speaker:
            return True
    return False


def transform(x: Row):
    if not check_speaker_in_entities(x.speaker, x.names):
        return None
    trie = create_trie(x.names)
    full_text = " ".join([x.leftContext, TARGET_QUOTE_TOKEN, x.rightContext])
    full_text = re.sub(r"\"+", "", full_text)
    try:
        masked_text, entities = find_entites(full_text, trie)
    except Exception:
        return None
    targets, ambiguous_flag = get_targets(entities, x.speaker)
    domain = x.domain if "domain" in x else ""
    pattern = x.pattern if "pattern" in x else ""

    return Row(
        articleUID=x.articleUID,
        articleOffset=x.articleOffset,
        speaker=x.speaker,
        quotation=x.quotation,
        full_text=full_text,
        masked_text=masked_text,
        entities=entities,
        targets=targets,
        ambiguous=ambiguous_flag,
        domain=domain,
        pattern=pattern,
    )


def transform_test(x: Row):
    trie = create_trie(x.names)
    full_text = " ".join([x.leftContext, TARGET_QUOTE_TOKEN, x.rightContext])
    full_text = re.sub(r"\"+", "", full_text)
    try:
        masked_text, entities = find_entites(full_text, trie)
    except Exception:
        return None

    return Row(
        articleUID=x.articleUID,
        articleOffset=x.articleOffset,
        quotation=x.quotation,
        full_text=full_text,
        masked_text=masked_text,
        entities=entities,
    )


@F.udf(returnType=BooleanType())
def is_all_lower(masked_text):
    text = re.sub(r'(\[MASK\]|\[QUOTE\]|\[TARGET_QUOTE\])', "", masked_text)
    return text == text.lower()


def extract_entities(
    spark: SparkSession,
    *,
    merged_path: str,
    speakers_path: str,
    output_path: str,
    nb_partition: int,
    compression: str = "gzip",
    ftype: str = "parquet",
    kind: str = "train",
):
    df = (
        spark.read.parquet(merged_path)
        if ftype == "parquet"
        else spark.read.json(merged_path).repartition(nb_partition)
    )
    df = df.dropna(subset=["quotation"])
    speakers = spark.read.json(speakers_path)
    joined = df.join(speakers, on="articleUID")

    if kind == "train":
        transformed = (
            joined.rdd.map(transform)
            .filter(lambda x: x is not None)
            .toDF()
            .withColumn("nb_targets", F.size("targets"))
            .withColumn("nb_entities", F.size("entities"))
            .filter("nb_entities > 0")
        )
        transformed.write.parquet(output_path, "overwrite", compression=compression)
    else:
        transformed = (
            joined.rdd.map(transform_test)
            .filter(lambda x: x is not None)
            .toDF()
            .withColumn("nb_entities", F.size("entities"))
            .filter("nb_entities > 0")
        )
        transformed.write.parquet(
            output_path, "overwrite", compression=compression
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--merged",
        type=str,
        help="Path to the merged output folder (.parquet), or raw quotes context (.json), in this case add --ftype json",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--speakers",
        type=str,
        help="Path to the speakers folder (.json)",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to output folder for the transformed data",
        required=True,
    )
    parser.add_argument(
        "--kind",
        type=str,
        help="Which kind of data it is to transform (train = with labels, test = without labels)",
        required=True,
        choices=["train", "test"],
    )
    parser.add_argument(
        "-l",
        "--local",
        help="Add if you want to execute locally. The code is expected to be run on a cluster if you run on big files",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--nb_partition",
        type=int,
        help="Number of partition for the output (useful if used with unsplittable compression algorithm). Default=50",
        default=200,
    )
    parser.add_argument(
        "--compression",
        type=str,
        help="Compression algorithm. Can be any compatible alogrithm with Spark Parquet. Default=gzip",
        default="gzip",
    )
    parser.add_argument(
        "--ftype",
        type=str,
        help="Filetype of the input data (json, parquet). Default=parquet",
        default="parquet",
    )
    args = parser.parse_args()

    if args.local:
        # import findspark

        # findspark.init()

        spark = (
            SparkSession.builder.master("local[24]")
            .appName("EntityExtractorLocal")
            .config("spark.driver.memory", "16g")
            .config("spark.executor.memory", "32g")
            .getOrCreate()
        )
    else:
        spark = SparkSession.builder.appName("EntityExtractor").getOrCreate()

    extract_entities(
        spark,
        merged_path=args.merged,
        speakers_path=args.speakers,
        output_path=args.output,
        nb_partition=args.nb_partition,
        compression=args.compression,
        ftype=args.ftype,
        kind=args.kind,
    )
