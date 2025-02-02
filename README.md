#  Quotebank: A Corpus of Quotations from a Decade of News

This repository contains the code for the following [paper](https://dl.acm.org/doi/10.1145/3437963.3441760), where we extracted Quotebank, a large corpus of annotated quotations. They were attributed using Quobert, our distantly and minimally supervised end-to-end, language-agnostic framework for quotation attribution.

> Timoté Vaucher, Andreas Spitz, Michele Catasta, and Robert West. 2021. *Quotebank: A Corpus of Quotations from a Decade of News*. In *Proceedings of the 14th ACM International Conference on Web Search and Data Mining* (WSDM '21). ACM, 2021.

## Dataset of attributed quotations [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4277311.svg)](https://doi.org/10.5281/zenodo.4277311)

Quotebank is a dataset of 178 million unique, speaker-attributed quotations that were extracted from 196 million English news articles crawled from over 377 thousand web domains between August 2008 and April 2020. Quotebank is available on [Zenodo](https://doi.org/10.5281/zenodo.4277311).

## Framework and reproducibility

### 0. Prerequisite
To run our code, you need:

- For the data pre-/postpreccing: A Spark (>= 2.3) cluster running Yarn, Python 3.x and Java 8
- For the training and inference: An instance w/ GPUs running on Python 3.7 with a venv described in `environment.yml`
  - *Note*: In the next steps, we don't include the steps were the data needs to be moved between HDFS and a local machine. A rule of thumb is that everything related to the models happens locally and every processing step in HDFS.
- To create the train data:
  - Your own dataset or the full Spinn3r dataset
  - Your own Wikidata people's dataset of the same format as our provided version `wikidata_people_ALIVE_FILTERED-NAMES-CLEAN.tsv.gz` that you can find in the [latest Release](https://github.com/epfl-dlab/Quotebank/releases/)
- Additionally, to create the evaluation data:
  - Our annotated data `annotated_mturk.json` in the [latest Release](https://github.com/epfl-dlab/Quotebank/releases/) 
- If you only want to run the inference step with our trained models:
  - The weights based on `bert-base-cased` of `quobert-base-cased` in the [latest Release](https://github.com/epfl-dlab/Quotebank/releases/)
  - The weights based on `bert-based-uncased` of `quobert-base-uncased` in the [latest Release](https://github.com/epfl-dlab/Quotebank/releases/)

### 1. Quotation and candidate extraction
The first step consists in extracting all direct quotations, their context and the candidate speakers from the data. More details about Quootstrap can be found in the [README of Quootstrap](quootstrap/README.md).
This can be generated with our variation Quootstrap by extracting the `quootstrap` tarball in the [latest Release](https://github.com/epfl-dlab/Quotebank/releases/) to get the required JARs and running the command `./extraction_quotations.sh` in your Spark cluster. It is important to verify the parameters in the `config.properties` file, i.e. you need to change `/path/to/` to suit your needs. Additionally, we want those parameters to be set to `True`:

```bash
EXPORT_RESULTS=true
DO_QUOTE_ATTRIBUTION=true

# Settings for exporting Article / Speakers
EXPORT_SPEAKERS=true

# Settings for exporting the quotes and context of the quotes
EXPORT_CONTEXT=true

# Optionally, we may need to export the articles too
EXPORT_ARTICLES=true
```


### 2. Data expansion and preparation for training
The next steps are in PySpark and the scripts are located under `dataprocessing/preprocessing`. We also provide a wrapper around `spark-submit` in `dataprocessing/run.sh`. Feel free to adapt it to your particular setup.

#### 2.1 Merging the Quotes-Context with Quootstrap output

This part is based on [`merge.py`](dataprocessing/preprocessing/merge.py), you can check the parameters to pass using `-h` option.

Run

```bash
./run.sh preprocessing/merge.py \
	-q /hadoop/path/output_quotebank \
	-c /hadoop/path/quotes_context \
	-o /hadoop/path/merged
```

#### 2.2 Getting the implicit contexts in remaining data 

This part is based on [`boostrap_EM.py`](quobert/dataprocessing/preprocessing/bootstrap_EM.py), you can check the parameters to pass using `-h` option. It finds the remaining contexts where quotations already attributed using Quootstrap have been found in implicit context.

Run

```bash
./run.sh preprocessing/boostrap_EM.py \
	-q /hadoop/path/output_quotebank \
	-c /hadoop/path/quotes_context \
	-o /hadoop/path/em_merged
```

#### 2.3 Extract the partial mentions of candidate entities in the data

This part is based on [`extract_entities.py`](quobert/dataprocessing/preprocessing/extract_entities.py), you can check the parameters to pass using `-h` option. It finds the partial mention of entities in the data from the full mentions extracted by Quootstrap. This is the last step before transforming the data into features for our model.

Run it for `merged` and `em_merged`

```bash
./run.sh preprocessing/extract_entities.py \
	-m /hadoop/path/merged \
	-s /hadoop/path/speakers \
	-o /hadoop/path/merged_transformed \
    --kind train

./run.sh preprocessing/extract_entities.py \
	-m /hadoop/path/em_merged \
	-s /hadoop/path/speakers \
	-o /hadoop/path/em_merged_transformed \
    --kind train
```

#### 2.4 (optional) Sample the data to deal with class imbalance

As we presented in the paper, our data is extremly imbalanced. We propose a sampling solution for both the case where the case is intact in [`sampling.py`](quobert/dataprocessing/preprocessing/sampling.py) and for the uncased case in [`sampling_uncased.py`](quobert/dataprocessing/preprocessing/sampling_uncased.py). You can check the parameters to pass using `-h` option. You probably need to adapt this script to the shape and size of your dataset.

Example for the cased case: Run the 2-step process

```bash
./run.sh preprocessing/sampling.py \
	--step generate \
	--path /hadoop/path/

./run.sh preprocessing/sampling.py \
	--step merge \
	--path /hadoop/path/
```

#### 2.5 Transform the data to features

This part is based on [`features.py`](quobert/dataprocessing/preprocessing/features.py), you can check the parameters to pass using `-h` option. If you don't want to use `bert-base-cased` based model, change the tokenizer here (`--tokenizer`)

Run:

```bash
./run.sh preprocessing/features.py \
	-t /hadoop/path/transformed \
	-o /hadoop/path/train_data
    --kind train
```

### 3. Model Training

The training of Quobert models is done in [`train.py`](train.py), you can check the parameters to pass using `-h` option. We assume you have a TensorBoard server running (e.g. in another `tmux` or `screen`)

You can for example run a training session using:
```bash
python train.py \
    --model_name_or_path bert-base-cased \
    --output_dir /path/to/model \
    --train_dir /path/to/train_data \
    --do_train
```
One could also evaluate the models on a validation set by setting `--do_eval` and `--eval_all_checkpoints` and passing a value to `--val_dir`

### 4. Model Testing on annotated data

To prepare the test data in `annotated_mturk.json`, repeat step *2.3* and *2.5* with this data. In step 2.3, additionally pass `--ftype json` as the test data is in json.

The evaluation of Quobert models on the annotated test set is done in [`test.py`](test.py), you can check the parameters to pass using `-h` option. We assume you have a TensorBoard server running (e.g. in another `tmux` or `screen`)

You can for example run a training session using:
```bash
python test.py \
    --model_dir /path/to/model \
    --output_dir /path/to/results \
    --test_dir /path/to/test_data
```

### 5. Inference and Postprocessing

#### 5.1 Preparing the inference data
As for testing, the data needs to be prepared before being fed to the model for inference. In our case, we used the `quotes_context` directly as input to step *2.3*:
```bash
./run.sh preprocessing/extract_entities.py \
    -m /hadoop/path/quotes_context \
    -s /hadoop/path/speakers \
    -o /hadoop/path/qc_transformed \
    --kind test
    --ftype json
```

Then proceed as in step *2.5*
```bash
./run.sh preprocessing/features.py \
	-t /hadoop/path/qc_transformed \
	-o /hadoop/path/inference_data
    --kind test
```

#### 5.2 Inference
The inference of data using Quobert models is done in [`inference.py`](inference.py), you can check the parameters to pass using `-h` option.

You can for example run a inference session using:
```bash
python inference.py \
    --model_dir /path/to/model \
    --output_dir /path/to/results \
    --inference_dir /path/to/inference_data
```

#### 5.3 (optional) Postprocessing the inference results
We also provide a pipeline to output the results in a format like [those made available to you on Zenodo](https://doi.org/10.5281/zenodo.4277311). The scripts for this steps are located under [`dataprocessing/postprocessing`](quobert/dataprocessing/postprocessing). It's a 2-step process, were we first find all the offsets of the candidate speakers mentioned in each article and then join the articles, quotes with their context, inference results and augmented speakers set.
```bash
./run.sh postprocessing/speakers_offset.py \
    -a /hadoop/path/articles \
    -s /hadoop/path/speakers \
    -o /hadoop/path/speakers_transformed

./run.sh postprocessing/process_res.py \
    -q /hadoop/path/quotes_context \
    -a /hadoop/path/articles \
    -s /hadoop/path/speakers_transformed \
    -r /hadoop/path/results \
    -o /hadoop/path/output
```


## Cite us

If you found the provided resources useful, please cite the above paper. Here's a BibTeX entry you may use:

```latex
@inproceedings{vaucher-2021-quotebank,
author = {Vaucher, Timot\'{e} and Spitz, Andreas and Catasta, Michele and West, Robert},
title = {Quotebank: A Corpus of Quotations from a Decade of News},
year = {2021},
isbn = {9781450382977},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3437963.3441760},
doi = {10.1145/3437963.3441760},
booktitle = {Proceedings of the 14th ACM International Conference on Web Search and Data Mining},
pages = {328–336},
numpages = {9},
keywords = {bert, quotation attribution, distant supervision, bootstrapping},
location = {Virtual Event, Israel},
series = {WSDM '21}
}
```

### Any questions or suggestions?

Contact [timote.vaucher@epfl.ch](mailto:timote.vaucher@epfl.ch).