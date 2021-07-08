import argparse
from string import punctuation

import pyspark.sql.functions as F
from pyspark.sql import SparkSession


@F.udf()
def process_quote(quote):
    return "".join(x for x in quote.lower() if x not in punctuation)


@F.udf()
def most_frequent(x):
    return max(set(x), key=x.count)


def find_exact_match(
    spark: SparkSession,
    quotes_context_path: str,
    quootstrap_path: str,
    output_path: str,
    nb_partition: int = 200,
    compression: str = "gzip",
):
    quootstrap_df = spark.read.json(quootstrap_path)
    quotes_context_df = spark.read.json(quotes_context_path)

    q2 = quootstrap_df.select(F.explode("occurrences").alias("occurrence"))
    fields_to_keep = [
        q2.occurrence.articleUID.alias("articleUID"),
        q2.occurrence.articleOffset.alias("articleOffset"),
    ]

    attributed_quotes_df = q2.select(*fields_to_keep)

    new_quotes_context_df = quotes_context_df.join(
        attributed_quotes_df, on=["articleUID", "articleOffset"], how="left_anti",
    )

    quootstrap_df.select(
        most_frequent("speaker").alias("speaker"),
        process_quote("quotation").alias("uncased_quote"),
    ).join(
        new_quotes_context_df.withColumn("uncased_quote", process_quote("quotation")),
        on="uncased_quote",
    ).drop(
        "uncased_quote"
    ).repartition(
        nb_partition
    ).write.parquet(
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
        "-l",
        "--local",
        help="Add if you want to execute locally.",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--nb_partition",
        type=int,
        help="Number of partition for the output (useful if used with unsplittable compression algorithm). Default=200",
        default=200,
    )
    parser.add_argument(
        "--compression",
        type=str,
        help="Compression algorithm. Can be any compatible alogrithm with Spark Parquet. Default=gzip",
        default="gzip",
    )
    args = parser.parse_args()

    print("Starting the Spark Session")
    if args.local:
        # import findspark

        # findspark.init()

        spark = (
            SparkSession.builder.master("local[24]")
            .appName("BootstrapLocal")
            .config("spark.driver.memory", "32g")
            .config("spark.executor.memory", "32g")
            .getOrCreate()
        )
    else:
        spark = SparkSession.builder.appName("Bootstrap_EM").getOrCreate()

    find_exact_match(
        spark,
        quootstrap_path=args.quootstrap,
        quotes_context_path=args.context,
        output_path=args.output,
        nb_partition=args.nb_partition,
        compression=args.compression,
    )
