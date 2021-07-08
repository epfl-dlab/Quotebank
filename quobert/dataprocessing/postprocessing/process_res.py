import argparse
import re
import string
from collections import Counter
from os.path import join

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType


@F.udf
def lower(text):
    no_punct = "".join(t for t in text if t not in string.punctuation)
    one_space = re.sub(r"\W+", " ", no_punct)
    return one_space.lower()


@F.udf
def longest(quotes):
    return sorted(quotes, key=len, reverse=True)[0]


@F.udf
def pad_int(idx):
    return f"{idx:06d}"


def first(col):
    return F.first(f"quotes.{col}").alias(col)


def get_col(col):
    return F.col(f"quotes.{col}").alias(col)


@F.udf(returnType=DoubleType())
def normalize(x):
    return x / 100.0 if x is not None else None


@F.udf
def get_top1(probas):
    return probas[0][0]


@F.udf(returnType=IntegerType())
def get_article_len(article):
    return len(article.split())


@F.udf
def get_most_common_name(names):
    return sorted(Counter(names).items(), key=lambda x: (-x[1], x[0]))[0][0]


@F.udf(returnType=ArrayType(StringType()))
def get_website(array):
    return list(dict.fromkeys(x["website"] for x in array))


@F.udf(returnType=ArrayType(StringType()))
def get_first_dedup(ids):
    return list({x[0] for x in ids})


NB_PARTITIONS = 2 ** 11
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--path", type=str, help="Root path to folder", required=True,
)
parser.add_argument('--partition', nargs='+', default=["year", "month"], help="Column to partition the out with. A choice of year and month. Order matters. Default 'year month'")

args = parser.parse_args()

spark = SparkSession.builder.appName("processRes").getOrCreate()
qc = spark.read.json(join(args.path, "quootstrap/quotes_context*")).repartition(
    NB_PARTITIONS
)
articles = spark.read.json(join(args.path, "quootstrap/articles*")).repartition(
    NB_PARTITIONS
)
speakers = spark.read.parquet(join(args.path, "quotebank/speakers")).repartition(
    NB_PARTITIONS
)
res = spark.read.csv(
    join(args.path, "quotebank/phase*"),
    header=True,
    schema="articleUID STRING, articleOffset LONG, rank INT, speaker STRING, proba DOUBLE",
).repartition(NB_PARTITIONS)

quote_article_link = (
    qc.join(articles.select("articleUID", "date", "website", "phase"), on="articleUID")
    .groupBy(lower(F.col("quotation")).alias("canonicalQuotation"))
    .agg(
        F.collect_set("quotation").alias("quotations"),
        F.min("date").alias("earliest_date"),
        F.min("phase").alias("phase"),
        F.count("*").alias("numOccurrences"),
        F.collect_list(
            F.struct(
                F.col("articleUID"),
                F.col("articleOffset"),
                F.col("quotation"),
                F.col("leftContext"),
                F.col("rightContext"),
                F.col("quotationOffset"),
                F.col("leftOffset").alias("contextStart"),
                F.col("rightOffset").alias("contextEnd"),
            )
        ).alias("quotes_link"),
        get_website(F.sort_array(F.collect_list(F.struct("date", "website")))).alias("urls"),
    )
    .withColumn("quotation", longest(F.col("quotations")))
    .withColumn(
        "row_nb",
        F.row_number().over(
            Window.partitionBy(F.to_date("earliest_date")).orderBy("canonicalQuotation")
        ),
    )
    .withColumn(
        "quoteID", F.concat_ws("-", F.to_date("earliest_date"), pad_int("row_nb")),
    )
    .withColumn("month", F.month("earliest_date"))
    .withColumn("year", F.year("earliest_date"))
    .drop("quotations", "row_nb")
)

joined_df = qc.join(res, on=["articleUID", "articleOffset"])

w = Window.partitionBy("canonicalQuotation")
rank_w = Window.partitionBy("canonicalQuotation").orderBy(F.desc("sum(proba)"))
agg_proba = (
    joined_df.groupBy(lower(F.col("quotation")).alias("canonicalQuotation"), "qids")
    .agg(F.sum("proba"), F.collect_list("speaker").alias("speakers"))
    .select(
        "*",
        F.sum("sum(proba)").over(w).alias("weight"),
        F.row_number().over(rank_w).alias("rank"),
        get_most_common_name("speakers").alias("speaker"),
    )
    .withColumn("proba", F.round(F.col("sum(proba)") / F.col("weight"), 4))
    .filter("proba >= 1e-4")
    .drop("sum(porba)", "weight")
)

agg_proba.write.parquet(
    join(args.path, "quotebank/quotebank-cache-proba"), mode="overwrite", compression="gzip"
)
agg_proba = spark.read.parquet(join(args.path, "quotebank/quotebank-cache-proba"))

top_speaker = agg_proba.filter("rank = 1").select(
    "canonicalQuotation",
    F.col("speaker").alias("top_speaker"),
    F.col("qids").alias("top_speaker_qid"),
    F.col("speakers").alias("top_surface_forms"),
)

probas = (
    agg_proba.orderBy("canonicalQuotation", "rank")
    .groupBy("canonicalQuotation")
    .agg(F.collect_list(F.struct(F.col("speaker"), F.col("proba"), F.col("qids"))).alias("probas"))
)

final = quote_article_link.join(top_speaker, on="canonicalQuotation").join(
    probas, on="canonicalQuotation"
)
final.write.parquet(
    join(args.path, "quotebank/quotebank-cache1"), mode="overwrite", compression="gzip"
)
final = spark.read.parquet(join(args.path, "quotebank/quotebank-cache1"))


SMALL_COLS = [
    "quoteID",
    "quotation",
    F.col("top_speaker").alias("speaker"),
    # F.col("top_speaker_qid").alias("qids"),
    F.col("earliest_date").alias("date"),
    "numOccurrences",
    "probas",
    "year",
    "month",
    "urls",
    "phase",
]

final.select(*SMALL_COLS).repartition(*args.partition).write.partitionBy(
    *args.partition
).json(
    join(args.path, "quotebank/quotes-df"), mode="overwrite", compression="bzip2",
)

BIG_COLS = [
    "quoteID",
    "numOccurrences",
    F.col("top_speaker").alias("globalTopSpeaker"),
    F.col("probas").alias("globalProbas"),
    F.explode("quotes_link").alias("quotes"),
]

individual_probas = (
    res.filter("proba > 0").orderBy("articleUID", "articleOffset", "rank")
    .groupBy("articleUID", "articleOffset")
    .agg(
        F.collect_list(
            F.struct("speaker", F.round(normalize("proba"), 4).alias("proba"), "qids")
        ).alias("localProbas")
    )
    .withColumn("speaker", get_top1("localProbas"))
)

df = final.select(*BIG_COLS)
df = df.join(
    individual_probas,
    on=[
        df.quotes.articleUID == individual_probas.articleUID,
        df.quotes.articleOffset == individual_probas.articleOffset,
    ],
).drop("articleUID", "articleOffset")

article_df = df.groupBy(F.col("quotes.articleUID").alias("articleUID")).agg(
    F.collect_list(
        F.struct(
            "quoteID",
            "numOccurrences",
            get_col("quotation"),
            get_col("quotationOffset"),
            get_col("contextStart"),
            get_col("contextEnd"),
            "globalTopSpeaker",
            "globalProbas",
            F.col("speaker").alias("localTopSpeaker"),
            "localProbas",
        )
    ).alias("quotations"),
)

article_df = (
    article_df.join(articles, on="articleUID")
    .join(speakers, on="articleUID")
    .withColumn("articleLength", get_article_len("content"))
    .withColumnRenamed("articleUID", "articleID")
    .withColumnRenamed("website", "url")
    .withColumn("year", F.year("date"))
    .withColumn("month", F.month("date"))
)

article_df.repartition(*args.partition).write.partitionBy(*args.partition).parquet(
    join(args.path, "quotebank/article-df"),
    mode="overwrite",
    compression="gzip",
)
