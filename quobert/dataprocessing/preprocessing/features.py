import argparse
import json
import re
import signal
import string
from typing import List, Optional, Union

from pyspark.broadcast import Broadcast
from pyspark.sql import Row, SparkSession
from transformers import AutoTokenizer

QUOTE_TOKEN = "[QUOTE]"
QUOTE_TARGET_TOKEN = "[TARGET_QUOTE]"
MASK_IDX = 103  # Index in BERT wordpiece

PUNCT = re.escape("".join(x for x in string.punctuation if x not in "[]"))


class TimeoutError(Exception):
    pass


def handler(signum, frame):
    raise TimeoutError()


def timeout(func, args=(), kwargs={}, timeout_duration=5, default=None):
    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError:
        print(f"Timeout Error running {func} with {args} and {kwargs}")
        result = default
    finally:
        signal.alarm(0)

    return result


def clean_text(masked_text):
    masked_text = re.sub(
        r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""",
        " ",
        masked_text,
    )
    masked_text = re.sub("[" + PUNCT + "]{4,}", " ", masked_text)
    return masked_text


def example_to_features(
    articleUID: str,
    articleOffset: int,
    masked_text: str,
    speaker: str,
    targets: List[int],
    entities,
    tokenizer: Union[AutoTokenizer, Broadcast],
    max_seq_len: int = 320,
    pad_to_max_length: bool = True,
) -> Optional[Row]:
    """Transform examples to QuoteFeatures row. Given the context and the speaker,
    extract the start/end offset that match the best the speaker.
    Those offsets will be used as targets for the models.

    Args:
        articleUID (str): The unique identifier of the associated article
        articleOffset (int): The offset (in number of quotes) in the associated article
        left_context (str): The text left of the quote
        quotation (str): The quote
        right_context (str): The text right of the quote
        speaker (str): The identified speaker
        tokenizer (BertTokenizer): The tokenizer to use to compute the features
        max_len_quotation (int, optional): Maximum length for the quotation in tokens. Defaults to 100.
        max_seq_len (int, optional): Maximum sequence length in tokens, extra tokens will be dropped. Defaults to 320.
        pad_to_max_length (bool, optional): Wheter to pad the tokens to `max_seq_len`. Pad on the right. Defaults to True.

    Returns:
        Optional[Row]: The features
    """
    tokenizer = tokenizer.value

    masked_text = timeout(clean_text, args=(masked_text,), default="")
    tokenized = timeout(tokenizer.tokenize, args=(masked_text,))
    if not tokenized or len(tokenized) > max_seq_len:
        # print(len(tokenized), masked_text)
        return None
    encoded = timeout(tokenizer.encode, args=(tokenized,))

    mask_idx = [0] + [
        i for i, idx in enumerate(encoded) if idx == MASK_IDX
    ]  # indexes of [CLS] and [MASK] token

    # if len(mask_idx) < 2:  # This should *NOT* happen
    #     # print("No mask token in", masked_text)
    #     return None

    return Row(
        uid=articleUID + " " + str(articleOffset),
        input_ids=encoded,
        mask_idx=mask_idx,
        target=targets[0],
        entities=json.dumps(entities),
        speaker=speaker,
    )


def transform_to_features(
    spark: SparkSession,
    *,
    transformed_path: str,
    tokenizer_model: str,
    output_path: str,
    nb_partition: int,
    compression: str,
    kind: str,
):
    """Entire transformation pipeline. Entry point to the process.
    Create a tokenizer from the model. Read the merged data and do the transformations.
    Finally write the resulting Dataset/Dataframe to the disk
    
    Args:
        merged_path (str): Path to the folder containing the merged data
        tokenizer_model (str): Model of the tokenizer. Must be supported by `transformers`
        output_path (str): Path to the output folder
        nb_partition (int): Number of partition for the output
        compression (str, optional): A parquet compatible compression algorithm. Defaults to 'gzip'.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    tokenizer.add_tokens([QUOTE_TOKEN, QUOTE_TARGET_TOKEN])
    tokenizer_bc = spark.sparkContext.broadcast(tokenizer)

    def __example_to_features(row: Row) -> Optional[Row]:
        return example_to_features(
            row.articleUID,
            row.articleOffset,
            row.masked_text,
            row.speaker if kind == "train" else "",
            row.targets if kind == "train" else [-1],
            row.entities,
            tokenizer_bc,
            pad_to_max_length=False,
        )

    transformed_df = spark.read.parquet(transformed_path)

    output_df = (
        transformed_df.rdd.repartition(2 ** 7)
        .map(__example_to_features)
        .filter(lambda x: x is not None)
        .toDF()
    )

    if kind == "test":
        output_df.drop("speaker", "target")  # .coalesce(nb_partition)
    output_df.write.parquet(output_path, "overwrite", compression=compression)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--transformed",
        type=str,
        help="Path to the transformed (sampled) output folder (.parquet)",
        required=True,
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name of the pretrained model, default: bert-base-cased",
        default="bert-base-cased",
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
    parser.add_argument(
        "--kind",
        type=str,
        help="Which kind of data it is to transform (train = with labels, test = without labels)",
        required=True,
        choices=["train", "test"],
    )
    args = parser.parse_args()

    print("Starting the Spark Session")
    if args.local:
        # import findspark

        # findspark.init()

        spark = (
            SparkSession.builder.master("local[24]")
            .appName("FeaturesExtractorLocal")
            .config("spark.driver.memory", "16g")
            .config("spark.executor.memory", "32g")
            .getOrCreate()
        )
    else:
        spark = SparkSession.builder.appName("FeaturesExtractor").getOrCreate()

    print("Starting the transformation to features")
    transform_to_features(
        spark,
        transformed_path=args.transformed,
        tokenizer_model=args.tokenizer,
        output_path=args.output,
        nb_partition=args.nb_partition,
        compression=args.compression,
        kind=args.kind,
    )
