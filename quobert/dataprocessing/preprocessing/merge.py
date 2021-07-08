import argparse
from urllib.parse import urlparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, to_date, udf


@udf
def get_domain(x):
    return urlparse(x).netloc


def merge(
    spark: SparkSession,
    *,
    quootstrap_path: str,
    quotes_context_path: str,
    output_path: str,
    nb_partition: int,
    compression: str = "gzip",
):
    """ Merge The output from Quootstrap together in order to create an input training set

    Args:
        spark (SparkSession): Current spark session
        quootstrap_path (str): HDFS path to the Quootstrap output (Q, S) pairs
        quotes_context_path (str): HDFS path to the quotes+context
        output_path (str): HDFS path to store the output of the merge
        nb_partition (int): Number of partition for the output
        compression (str, optional): A parquet compatible compression algorithm. Defaults to 'gzip'.
    """
    quootstrap_df = spark.read.json(quootstrap_path)
    quotes_context_df = spark.read.json(quotes_context_path)

    # Extract all quotes and speakers from Quootstrap data
    q2 = quootstrap_df.select("quotation", explode("occurrences").alias("occurrence"))

    fields_to_keep = [
        q2.occurrence.articleUID.alias("articleUID"),
        q2.occurrence.articleOffset.alias("articleOffset"),
        q2.occurrence.matchedSpeakerTokens.alias("speaker"),
        q2.occurrence.extractedBy.alias("pattern"),
        get_domain(q2.occurrence.website).alias("domain"),
        to_date(q2.occurrence.date).alias("date"),
    ]

    attributed_quotes_df = q2.select(*fields_to_keep)

    # Merge df and write parquet files
    attributed_quotes_context_df = attributed_quotes_df.join(
        quotes_context_df, on=["articleUID", "articleOffset"], how="inner"
    )

    attributed_quotes_context_df.repartition(nb_partition).write.parquet(
        output_path, "overwrite", compression=compression
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q",
        "--quootstrap",
        type=str,
        help="Path to Quoostrap output (.json)",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--context",
        type=str,
        help="Path to folder with all quotes with context (.json.gz)",
        required=True,
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Path to output folder", required=True
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
        "-l",
        "--local",
        help="Add if you want to execute locally. The code is expected to be run on a cluster if you run on big files",
        action="store_true",
    )
    args = parser.parse_args()

    print("Starting the Spark Session")
    if args.local:
        import findspark

        findspark.init()

        spark = (
            SparkSession.builder.master("local[24]")
            .appName("QuoteMergerLocal")
            .config("spark.driver.memory", "16g")
            .config("spark.executor.memory", "32g")
            .getOrCreate()
        )
    else:
        spark = SparkSession.builder.appName("QuoteMerger").getOrCreate()

    print("Starting the merging process")
    merge(
        spark,
        quootstrap_path=args.quootstrap,
        quotes_context_path=args.context,
        output_path=args.output,
        nb_partition=args.nb_partition,
        compression=args.compression,
    )
