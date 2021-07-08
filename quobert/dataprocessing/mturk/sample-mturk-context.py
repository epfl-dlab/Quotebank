import argparse
import os

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


def extract(
    spark: SparkSession,
    *,
    mturk_path: str,
    output_path: str,
    nb_partition: int,
    compression: str = "gzip",
):
    df = spark.read.parquet(mturk_path)
    df_key = df.withColumn(
        "key", F.substring(df.articleUID, 1, 6).cast("int")
    )
    key_proportions = df_key.groupBy("key").count().orderBy("key").collect()
    proportions = {r.key: 100 / r["count"] for r in key_proportions}
    selected_keys = df_key.sampleBy("key", proportions, seed=1).withColumn("source", F.lit("normal")).cache()
    print("number of normal quotation", selected_keys.count())

    speaker_proportions = df_key.groupBy("nb_entities").count().orderBy("nb_entities").collect()
    proportions = {r.nb_entities: min(1., 100 / r["count"]) if r.nb_entities <= 20 else 0 for r in speaker_proportions}
    selected_speakers = df_key.sampleBy("nb_entities", proportions, seed=2).withColumn("source", F.lit("nb_entities")).cache()
    print("number of quotation by more speakers", selected_speakers.count())

    df_filtered = df_key.filter(~df_key.in_quootstrap)
    nb_hard = df_filtered.count()
    selected_hard = df_filtered.sample(fraction=5000 / nb_hard, seed=3).withColumn("source", F.lit("not_quootstrap")).cache()
    print("number of quotation not in quootstrap", selected_hard.count())
    to_evaluate = selected_keys.union(selected_speakers).union(selected_hard).dropDuplicates(['articleUID', 'articleOffset']).dropDuplicates(['context']).cache()
    print("total after dropDuplicates", to_evaluate.count())
    to_evaluate.coalesce(nb_partition).write.parquet(
        os.path.join(output_path, "mturk"),
        mode="overwrite",
        compression=compression,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mturk",
        type=str,
        help="Path to folder with all raw context for mturk (.gz.parquet)",
        required=True,
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Path to output folder", required=True
    )
    parser.add_argument(
        "-n",
        "--nb_partition",
        type=int,
        help="Number of partition for the output (useful if used with unsplittable compression algorithm). Default=10",
        default=10,
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
            .appName("SampleMTurkLocal")
            .config("spark.driver.memory", "16g")
            .config("spark.executor.memory", "32g")
            .getOrCreate()
        )
    else:
        spark = SparkSession.builder.appName("SampleMTurk").getOrCreate()

    print("Starting the merging process")
    extract(
        spark,
        mturk_path=args.mturk,
        output_path=args.output,
        nb_partition=args.nb_partition,
        compression=args.compression,
    )
