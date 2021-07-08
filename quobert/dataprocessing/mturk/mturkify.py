import argparse
import re
import string

import pyspark.sql.functions as F
from pyspark.sql import Row, SparkSession

from pygtrie import StringTrie

PUNCT = "".join(x for x in string.punctuation if x not in "[]")
TARGET_QUOTE_TOKEN = "[TARGET_QUOTE]"
QUOTE_TOKEN = "[QUOTE]"
MASK_TOKEN = "[MASK]"
BLACK_LIST = {"president", "manager"}


def create_trie(names):
    trie = StringTrie(delimiter="/")
    for (name, qid) in names:
        q_name = eval(qid[0])[1]
        name = [x for x in name if x.lower() not in BLACK_LIST]
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


def add_bold(text):
    return f"<b>{text}</b>"


def add_highlight(text):
    return f"<mark>&ldquo;{text}&rdquo;</mark>"


def get_entity(entites):
    if not type(entites) == tuple:
        raise Exception(f"{entites} is not a tuple")
    return entites[0]


def find_entites(text: str, trie: StringTrie, mask: str = MASK_TOKEN):
    tokens = text.split()
    tokens = fix_punct_tokens(tokens)
    start = 0
    count = 1  # start at 1, 0 is for the "NO_MATCH"
    entities = dict()
    out = []
    for i in range(len(tokens)):
        key = "/".join(tokens[start : i + 1]).lower()
        # name = " ".join(tokens[start: i + 1])
        if trie.has_subtrie(key):  # Not done yet
            if i == len(tokens) - 1:  # Reached the end of the string
                entities[count] = get_partial_match(trie, key)
                out.append(add_bold(get_entity(entities[count])))
        elif trie.has_key(key):  # noqa: W601  # Find a perfect match
            entities[count] = trie[key]
            out.append(add_bold(get_entity(entities[count])))
            count += 1
            start = i + 1
        elif start < i:  # Found partial prefix match before this token
            old_key = "/".join(tokens[start:i]).lower()
            #  name = " ".join(tokens[start:i])
            entities[count] = get_partial_match(trie, old_key)
            out.append(add_bold(get_entity(entities[count])))
            count += 1
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


def transform(x: Row):
    trie = create_trie(x.names)
    full_text = " ".join([x.leftContext, TARGET_QUOTE_TOKEN, x.rightContext])
    full_text = re.sub(r"\"+", "", full_text)
    try:
        context, entities = find_entites(full_text, trie)
        context = re.sub(
            re.escape(TARGET_QUOTE_TOKEN), add_highlight(x.quotation), context
        )
        context = re.sub(
            re.escape(QUOTE_TOKEN), "&ldquo;[unrelated quotation]&rdquo;", context
        )
    except Exception:
        #  print("Problem with this context:", context, "with error", e)
        return None

    return Row(
        articleUID=x.articleUID,
        articleOffset=x.articleOffset,
        context=context,
        entities=list(entities.keys()),
    )


def mturkify(
    spark: SparkSession,
    *,
    quotes_context_path: str,
    quootstrap_path: str,
    speakers_path: str,
    output_path: str,
    nb_partition: int,
    compression: str = "gzip",
):
    qc = spark.read.json(quotes_context_path)
    qc_date = qc.withColumn(
        "year", F.substring(qc.articleUID, 1, 4).cast("int")
    ).withColumn("month", F.substring(qc.articleUID, 5, 2).cast("int"))
    qc_filtered = qc_date.filter("year > 2014 OR (year = 2014 AND month >= 6)")
    qc_key = qc_filtered.withColumn(
        "key", F.substring(qc_filtered.articleUID, 1, 6).cast("int")
    ).drop("year", "month")

    quootstrap_df = spark.read.json(quootstrap_path)
    q2 = quootstrap_df.select(F.explode("occurrences").alias("occurrence"))
    fields_to_keep = [
        q2.occurrence.articleUID.alias("articleUID"),
        q2.occurrence.articleOffset.alias("articleOffset"),
    ]

    attributed_quotes_df = q2.select(*fields_to_keep).withColumn(
        "in_quootstrap", F.lit(True)
    )

    speakers = spark.read.json(speakers_path)
    joined = qc_key.join(speakers, on="articleUID")
    transformed = (
        joined.rdd.map(transform)
        .filter(lambda x: x is not None)
        .toDF()
        .withColumn("nb_entities", F.size("entities"))
    )

    transformed_quootstrap = transformed.join(
        attributed_quotes_df, on=["articleUID", "articleOffset"], how="left"
    ).na.fill(False)
    transformed_quootstrap.write.parquet(
        output_path, "overwrite", compression=compression
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--context",
        type=str,
        help="Path to folder with all quotes with context (.json.gz)",
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
        "-q",
        "--quootstrap",
        type=str,
        help="Path to the quootstrap output folder (.json)",
        required=True,
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Path to output folder", required=True
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
        default=50,
    )
    parser.add_argument(
        "--compression",
        type=str,
        help="Compression algorithm. Can be any compatible alogrithm with Spark Parquet. Default=gzip",
        default="gzip",
    )
    args = parser.parse_args()

    if args.local:
        # import findspark

        # findspark.init()

        spark = (
            SparkSession.builder.master("local[24]")
            .appName("MturkifyLocal")
            .config("spark.driver.memory", "16g")
            .config("spark.executor.memory", "32g")
            .getOrCreate()
        )
    else:
        spark = SparkSession.builder.appName("Mturkify").getOrCreate()

    mturkify(
        spark,
        quotes_context_path=args.context,
        quootstrap_path=args.quootstrap,
        speakers_path=args.speakers,
        output_path=args.output,
        nb_partition=args.nb_partition,
        compression=args.compression,
    )
