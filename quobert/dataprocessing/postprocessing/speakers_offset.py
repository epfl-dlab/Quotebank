import argparse
import string

from pyspark.sql import Row, SparkSession

from pygtrie import StringTrie  # type: ignore # Sideloaded in the spark-submit

PUNCT = "".join(x for x in string.punctuation if x not in "[]")
BLACK_LIST = {"president", "manager"}


def create_trie(names):
    trie = StringTrie(delimiter="/")
    for name in names:
        processed_name = [
            x for x in name["name"].split() if x.lower() not in BLACK_LIST
        ]
        for i in range(len(processed_name)):
            trie["/".join(processed_name[i:]).lower()] = (name["name"], name["ids"])
    return trie


def update_entity(name, qinfo, start, end, out):
    if name in out:
        out[name]["offsets"].append([start, end])
    else:
        out[name] = {
            "name": name,
            "ids": sorted({x[0] for x in qinfo}),
            "offsets": [[start, end]],
        }


def reduce_entities(entities):
    out = dict()
    for i, (value, start, end) in entities.items():
        if isinstance(value, list):
            for name, qinfo in value:
                update_entity(name, qinfo, start, end, out)
        else:
            try:
                update_entity(value[0], value[1], start, end, out)
            except IndexError:  # this happens when value = '/', which is rare
                print(value, "We had an error here")
    return list(out.values())


def get_partial_match(trie, key):
    match = list(trie[key:])
    return match if len(match) > 1 else match[0]


def get_entity(trie, key):
    entites = get_partial_match(trie, key)
    if not type(entites) == tuple:
        raise Exception(f"{entites} is not a tuple")
    return entites


def find_entites(text: str, trie: StringTrie):
    tokens = text.split()
    start = 0
    count = 1  # start at 1, 0 is for the "NO_MATCH"
    entities = dict()
    for i in range(len(tokens)):
        key = "/".join(tokens[start : i + 1]).lower()
        if trie.has_subtrie(key):  # Not done yet
            if i == len(tokens) - 1:  # Reached the end of the string
                entities[count] = (get_entity(trie, key), start, i + 1)
        elif trie.has_key(key):  # noqa: W601 # Find a perfect match
            entities[count] = (trie[key], start, i + 1)
            count += 1
            start = i + 1
        elif start < i:  # Found partial prefix match before this token
            old_key = "/".join(tokens[start:i]).lower()
            entities[count] = (get_entity(trie, old_key), start, i)
            count += 1
            if trie.has_node(
                tokens[i].lower()
            ):  # Need to verify that the current token isn't in the Trie
                start = i
            else:
                start = i + 1
        else:  # No match
            start = i + 1
    return reduce_entities(entities)


def transform(x: Row):
    trie = create_trie(x.names)
    try:
        entities = find_entites(x.content, trie)
    except Exception:
        return None

    return Row(articleUID=x.articleUID, names=entities,)


def speakers_offset(
    spark: SparkSession,
    *,
    articles_path: str,
    speakers_path: str,
    output_path: str,
    nb_partition: int,
    compression: str = "gzip",
):
    df = (
        spark.read.json(articles_path)
        .select("articleUID", "content")
        .repartition(nb_partition)
    )
    speakers = spark.read.json(speakers_path)
    joined = df.join(speakers, on="articleUID")

    transformed = joined.rdd.map(transform).filter(lambda x: x is not None).toDF()
    transformed.write.parquet(output_path, "overwrite", compression=compression)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--articles",
        type=str,
        help="Path to the articles (.json)",
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
        help="Path to output folder for the transformed speaker offsets",
        required=True,
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
    args = parser.parse_args()

    if args.local:
        # import findspark

        # findspark.init()

        spark = (
            SparkSession.builder.master("local[24]")
            .appName("SpeakerOffsetsLocal")
            .config("spark.driver.memory", "16g")
            .config("spark.executor.memory", "32g")
            .getOrCreate()
        )
    else:
        spark = SparkSession.builder.appName("SpeakerOffsets").getOrCreate()

    speakers_offset(
        spark,
        articles_path=args.articles,
        speakers_path=args.speakers,
        output_path=args.output,
        nb_partition=args.nb_partition,
        compression=args.compression,
    )
