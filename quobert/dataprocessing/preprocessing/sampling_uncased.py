# Sample and then merge examples where the data is in lower case

import argparse
import sys
import re
from os.path import join

from pyspark.sql import SparkSession, Window, Row
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType, BooleanType

SEED = 42


@F.udf(returnType=BooleanType())
def is_all_lower(masked_text):
    text = re.sub(r"(\[MASK\]|\[QUOTE\]|\[TARGET_QUOTE\])", "", masked_text)
    return text == text.lower()


def filter_df(spark, path):
    merged = spark.read.parquet(join(path, "train/merged_transformed/"))
    em = spark.read.parquet(join(path, "train/EM_transformed/"))

    merge_rand = merged.filter(is_all_lower("masked_text")).withColumn(
        "rand", F.rand(seed=SEED)
    )
    em_rand = em.filter(is_all_lower("masked_text")).withColumn(
        "rand", F.rand(seed=SEED)
    )

    merge_nambiguous = merge_rand.filter(~merge_rand.ambiguous)
    em_nambiguous = em_rand.filter(~em_rand.ambiguous)
    merge_ambiguous = merge_rand.filter(merge_rand.ambiguous)
    em_ambiguous = em_rand.filter(em_rand.ambiguous)

    return merge_nambiguous, em_nambiguous, merge_ambiguous, em_ambiguous


def create_neg_example(row):
    entities = dict(row.entities)
    try:
        del entities[row.speaker]
    except KeyError:
        print(row.speaker, "not in", row.entities, row.masked_text, file=sys.stderr)
        return None
    cur_target = row.targets[0]
    new_masked_text = []
    i = 0
    for token in row.masked_text.split():
        if token == "[MASK]":
            i += 1
            if cur_target == i:
                new_masked_text.append(row.speaker)
            else:
                new_masked_text.append(token)
        else:
            new_masked_text.append(token)
    return Row(
        articleUID=row.articleUID,
        articleOffset=row.articleOffset,
        full_text=row.full_text,
        masked_text=" ".join(new_masked_text),
        quotation=row.quotation,
        entities=entities,
        targets=[0],
        speaker=row.speaker,
    )


def create_subsample(spark, path):
    merge_nambiguous, em_nambiguous, merge_ambiguous, em_ambiguous = filter_df(
        spark, path
    )

    w_pattern = Window.partitionBy("pattern")
    w_entity = Window.partitionBy("nb_entities")
    w_domain = Window.partitionBy("domain")
    w_pattern_entity = Window.partitionBy("pattern", "nb_entities")

    # UNAMBIGUOUS DATA
    merge_w = merge_nambiguous.select(
        "*",
        F.count("*").over(w_pattern).alias("pattern_count"),
        F.count("*").over(w_domain).alias("domain_count"),
    ).select(
        "articleOffset",
        "articleUID",
        "full_text",
        "masked_text",
        "quotation",
        "entities",
        "speaker",
        "targets",
        "rand",
        F.when(F.col("domain_count") >= 100, F.col("domain"))
        .otherwise("others")
        .alias("domain"),
        F.when(F.col("nb_entities") <= 20, F.col("nb_entities"))
        .otherwise(21)
        .alias("nb_entities"),
        F.when(F.col("pattern_count") >= 500, F.col("pattern"))
        .otherwise("others")
        .alias("pattern"),
    )

    @F.udf(returnType=FloatType())
    def get_proba(nb_samples, max_samples=400):
        return min(1.0, max_samples / nb_samples)

    subsample = (
        merge_w.select("*", F.count("*").over(w_pattern_entity).alias("pe_count"))
        .withColumn("proba", get_proba("pe_count"))
        .filter("rand <= proba")
        .drop("rand", "pe_count", "proba")
    )

    subsample_pos, subsample_neg = subsample.randomSplit([0.8, 0.2], seed=SEED)

    subsample_pos.coalesce(32).write.parquet(
        join(path, "sampling/quootstrap_subsample_lower"),
        "overwrite",
        compression="gzip",
    )

    subsample_neg.rdd.map(create_neg_example).filter(
        lambda x: x is not None
    ).toDF().write.parquet(
        join(path, "sampling/quootstrap_subsample_neg_lower"),
        "overwrite",
        compression="gzip",
    )

    neg_examples = (
        em_nambiguous.select("*", F.explode("targets").alias("target"))
        .filter(F.col("target") == 0)
        .drop("target")
    )
    neg_examples.write.parquet(
        join(path, "sampling/neg_examples_lower"),
        "overwrite",
        compression="gzip",
    )

    em_nambiguous_target = em_nambiguous.join(
        neg_examples, on=["articleUID", "articleOffset"], how="leftanti"
    )
    em_w = em_nambiguous_target.select(
        "*", F.count("*").over(w_entity).alias("entities_count")
    ).select(
        "articleOffset",
        "articleUID",
        "full_text",
        "masked_text",
        "quotation",
        "entities",
        "speaker",
        "targets",
        "rand",
        "entities_count",
        F.when(F.col("nb_entities") <= 20, F.col("nb_entities"))
        .otherwise(21)
        .alias("nb_entities"),
    )

    @F.udf(returnType=FloatType())
    def get_proba_bis(nb_samples, max_samples=220_000):
        return min(1.0, max_samples / nb_samples)

    em_subsample = (
        em_w.withColumn("proba", get_proba_bis("entities_count"))
        .filter("rand <= proba")
        .drop("rand", "entities_count", "proba")
    )

    em_subsample_pos, em_subsample_neg = em_subsample.randomSplit([0.8, 0.2], seed=SEED)

    em_subsample_pos.write.parquet(
        join(path, "sampling/em_subsample_lower"),
        "overwrite",
        compression="gzip",
    )

    em_subsample_neg.rdd.map(create_neg_example).filter(
        lambda x: x is not None
    ).toDF().write.parquet(
        join(path, "sampling/em_subsample_neg_lower"),
        "overwrite",
        compression="gzip",
    )


def merge_subsample(spark, path):
    subsample = spark.read.parquet(
        join(path, "sampling/quootstrap_subsample_lower")
    )
    subsample_neg = spark.read.parquet(
        join(path, "sampling/quootstrap_subsample_neg_lower")
    )
    neg_examples = spark.read.parquet(
        join(path, "sampling/neg_examples_lower")
    )
    em_subsample = spark.read.parquet(
        join(path, "sampling/em_subsample_lower")
    )
    em_subsample_neg = spark.read.parquet(
        join(path, "sampling/em_subsample_neg_lower")
    )
    # subsample_ambiguous = spark.read.parquet(
    #     join(path, "sampling/quootstrap_ambiguous_subsample")
    # )
    # em_subsample_ambiguous = spark.read.parquet(
    #     join(path, "sampling/em_ambiguous_subsample")
    # )

    COL_TO_KEEP = [
        "articleUID",
        "articleOffset",
        "full_text",
        "masked_text",
        "quotation",
        "entities",
        "targets",
        "speaker",
    ]

    VALIDATION_RATIO = 0.01
    QUOOTSTRAP = 300_000
    NON_QUOOTSTRAP = 490_000
    NEG = 250_000

    train_subsample, val_subsample = subsample.sample(
        fraction=QUOOTSTRAP / subsample.count(), seed=SEED
    ).randomSplit([1 - VALIDATION_RATIO, VALIDATION_RATIO], SEED)
    train_subsample_neg, val_subsample_neg = subsample_neg.sample(
        fraction=NEG * 0.3 / (subsample_neg.count()), seed=SEED
    ).randomSplit([1 - VALIDATION_RATIO, VALIDATION_RATIO], SEED)
    train_em, val_em = em_subsample.sample(
        fraction=NON_QUOOTSTRAP / em_subsample.count(), seed=SEED
    ).randomSplit([1 - VALIDATION_RATIO, VALIDATION_RATIO], SEED)
    train_em_neg, val_em_neg = em_subsample_neg.sample(
        fraction=NEG * 0.3 / (em_subsample_neg.count()), seed=SEED
    ).randomSplit([1 - VALIDATION_RATIO, VALIDATION_RATIO], SEED)
    train_neg, val_neg = neg_examples.sample(
        fraction=NEG * 0.4 / neg_examples.count(), seed=SEED
    ).randomSplit([1 - VALIDATION_RATIO, VALIDATION_RATIO], SEED)

    # val_ambiguous_subsample = subsample_ambiguous.sample(
    #     fraction=VALIDATION_SIZE / subsample_ambiguous.count(), seed=SEED
    # )
    # val_ambiguous_em = em_subsample_ambiguous.sample(
    #     fraction=VALIDATION_SIZE / em_subsample_ambiguous.count(), seed=SEED
    # )

    train_set = (
        train_subsample.select(*COL_TO_KEEP)
        .union(train_em.select(*COL_TO_KEEP))
        .union(train_neg.select(*COL_TO_KEEP))
        .union(train_subsample_neg.select(*COL_TO_KEEP))
        .union(train_em_neg.select(*COL_TO_KEEP))
    )

    # val_set = (
    #     val_subsample.select(*COL_TO_KEEP)
    #     .union(val_em.select(*COL_TO_KEEP))
    #     .union(val_neg.select(*COL_TO_KEEP))
    #     .union(val_ambiguous_subsample.select(*COL_TO_KEEP))
    #     .union(val_ambiguous_em.select(*COL_TO_KEEP))
    #     .union(val_em_neg.select(*COL_TO_KEEP))
    #     .union(val_subsample_neg.select(*COL_TO_KEEP))
    # )

    train_set.write.parquet(
        join(path, "sampling/train_set_empirical_lower"),
        mode="overwrite",
        compression="gzip",
    )
    # val_set.write.parquet(join(path, "sampling/val_set"), mode="overwrite", compression="gzip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--step",
        type=str,
        help="Which step to run",
        required=True,
        choices=["generate", "merge"],
    )
    parser.add_argument(
        "-p", "--path", type=str, help="root path to folder", required=True,
    )

    args = parser.parse_args()
    spark = SparkSession.builder.appName("SamplingLower").getOrCreate()

    if args.step == "generate":
        create_subsample(spark, args.path)
    elif args.step == "merge":
        merge_subsample(spark, args.path)
