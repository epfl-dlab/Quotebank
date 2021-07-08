# Quootstrap
This folder contains an adapted version of the reference implementation of Quootstrap, as described in the paper [[PDF]](https://dlab.epfl.ch/people/west/pub/Pavllo-Piccardi-West_ICWSM-18.pdf):
> Dario Pavllo, Tiziano Piccardi, Robert West. *Quootstrap: Scalable Unsupervised Extraction of Quotation-Speaker Pairs from Large News Corpora via Bootstrapping*. In *Proceedings of the 12th International Conference on Web and Social Media (ICWSM),* 2018.

#### Disclaimer

*This folder contains an improved version of the code from the paper using Wikidata instead of Freebase for entity linking and with the option of saving all the quotes with their context and speakers, instead of only those extracted by the rules. For the code of the paper, please check [this branch]( https://github.com/epfl-dlab/quootstrap/tree/master )*

## How to run

Go to the **Release** section and download the .zip archive, which contains the executable `quootstrap.jar` as well as all necessary dependencies and configuration files. You can also find a convenient script `extraction_quotations.sh` that can be used to run the application on a Yarn cluster. The script runs this command:
```bash
spark-submit --jars opennlp-tools-1.9.2.jar,spinn3r-client-3.4.05-edit.jar,stanford-corenlp-3.8.0.jar,jsoup-1.10.3.jar,guava-14.0.1.jar \
	--num-executors 25 \
	--executor-cores 16 \
	--driver-memory 128g \
	--executor-memory 128g \
	--conf "spark.executor.memoryOverhead=32768" \
	--class ch.epfl.dlab.quootstrap.QuotationExtraction \
	--master yarn \
	quootstrap.jar
```
After tuning the settings to suit your particular configuration, you can run the command as:
```bash
./extraction_quotations.sh
```

### Setup

To run our code, you need:
- Java 8
- Spark 2.3
- The entire Spinn3r dataset (available on the hadoop cluster) or your own dataset
- Our dataset of people extracted from Wikidata (available on `/data`)

### How to build

Clone the repository and import it as an Eclipse project. All dependencies are downloaded through Maven. To build the application, generate a .jar file with all source files and run it as explained in the previous section. Alternatively, you can use Spark in local mode for experimenting. Additional instructions on how to extend the project with new functionalities (e.g. support for new datasets) are reported later.

## Configuration
The first configuration file is `config.properties`. The most important fields in order to get the application running are:
- `NEWS_DATASET_PATH` specifies the HDFS path of the Spinn3r news dataset
- `PEOPLE_DATASET_PATH` specifies the HDFS path of the Wikidata people list.
- `NEWS_DATASET_LOADER` specifies which loader to use for the dataset.
- `EXPORT_PATH` specifies the HDFS path for the (quotation, speaker) pairs output.
- `CONTEXT_PATH` specifies the HDFS path for the (quotation, context, lang) output.
- `NUM_ITERATIONS=1` specifies the number of iterations of the extraction algorithm. Set it to 1 if you want to run the algorithm only on the seed patterns (iteration 0). A number of iterations between 3 and 5 is more than enough. **Note:** Currently this feature is not supported anymore and a fix to support wikidata would be appraciated.

Additionally you can change the flow of the application with the following parameters:

- `DO_QUOTATION_ATTRIBUTION` & `EXPORT_RESULTS`: Whether to perform Quote Attribution and export the results. This is the original output from Quootstrap.
- `EXPORT_CONTEXT`: Whether to export all the quotes and their context. Not present in the original version of Quootstrap
- `EXPORT_SPEAKERS`: Whether to export all full matches of candidate in a given article. Partial matches are dealt with in Quobert
- `EXPORT_ARTICLE`: Whether to export all articles in a easier to read format (not recommanded if your input data is already easily readable)

The second configuration file is `seedPatterns.txt`, which, as the name suggests, contains the seed patterns that are used in the first iteration, one by line.


## Exporting results
### Quotation-speaker pairs

You can save the results as a HDFS text file formatted in JSON, with one record per line. For each record, the full quotation is exported, as well as the full name of the speaker (as reported in the article), his/her Wikidata ID, the confidence value of the tuple, and the occurrences in which the quotation was found. As for the latter, we report the article ID, an incremental offset within the article (which is useful for linking together split quotations), the pattern that extracted the tuple along with its confidence, the website, and the date the article appeared.

```json
{
  "canonicalQuotation":"action is easy it is very easy comedy is difficult",
  "confidence":1.0,
  "numOccurrences":6,
  "numSpeakers":1,
  "occurrences":[
    {"articleOffset":0,
     "articleUID":"2012031008_00073468_W",
     "date":"2012-03-10 08:49:40",
     "extractedBy":"$Q , $S said",
     "matchedSpeakerTokens":"Akshay Kumar",
     "patternConfidence":1.0,
     "quotation":"action is easy, it is very easy. comedy is difficult,",
     "website":"http://deccanchronicle.com/channels/showbiz/bollywood/action-easy-comedy-toughest-akshay-kumar-887"},
    ...
    {"articleOffset":0,
     "articleUID":"2012031012_00038989_W",
     "date":"2012-03-10 12:23:40",
     "extractedBy":"$Q , $S told",
     "matchedSpeakerTokens":"Akshay Kumar",
     "patternConfidence":0.7360332115513023,
     "quotation":"action is easy, it is very easy. comedy is difficult,",
     "website":"http://hindustantimes.com/Entertainment/Bollywood/Comedy-is-difficult-Akshay-Kumar/Article1-823315.aspx"}
  ],
  "quotation":"action is easy, it is very easy. comedy is difficult,",
  "speaker":["Akshay Kumar"],
  "speakerID":["Q233748"]
}
```
Remarks:
- *articleOffset* can have gaps (but quotations are guaranteed to be ordered correctly).
- *canonicalQuotation* is the internal representation of a particular quotation, used for pattern matching purposes. The string is converted to lower case and punctuation marks are removed.
- As in the example above, the full quotation might differ from the one(s) found in *occurrences* due to the quotation merging mechanism. We always report the longest (and most likely useful) quotation when multiple choices are possible.

## Adding support for new datasets/formats

If you want to add support for other datasets/formats, you can provide a concrete implementation for the Java interface `DatasetLoader` and specify its full class name in the `NEWS_DATASET_LOADER` field of the configuration. For each article, you must supply a unique ID (int64/long), the website in which it can be found, and its content in tokenized format, i.e. as a list of strings. We provide an implementation for our JSON Spinn3r dataset in `ch.epfl.dlab.quootstrap.Spinn3rDatasetLoader`. For parquet dataframes in `ch.epfl.dlab.quootstrap.ParquetDatasetLoader`, and for Standford Spinn3r data format `ch.epfl.dlab.quootstrap.Spinn3rTextDatasetLoader`.

## Replacing the tokenizer
If, for any reason (e.g. license, language other than English), you do not want to depend on Stanford PTBTokenizer, you can provide your own implementation of the `ch.epfl.dlab.spinn3r.Tokenizer` interface. You only have to implement two methods: `tokenize` and `untokenize`. Tokenization is one of the least critical steps in our pipeline, and does not impact the final result significantly.

## License
We release our work under the MIT license. Third-party components, such as Stanford CoreNLP, are subject to their respective licenses.

If you use our code and/or data in your research, please cite our paper [[PDF]](https://dlab.epfl.ch/people/west/pub/Pavllo-Piccardi-West_ICWSM-18.pdf):
```
@inproceedings{quootstrap2018,
  title={Quootstrap: Scalable Unsupervised Extraction of Quotation-Speaker Pairs from Large News Corpora via Bootstrapping},
  author={Pavllo, Dario and Piccardi, Tiziano and West, Robert},
  booktitle={Proceedings of the 12th International Conference on Web and Social Media (ICWSM)},
  year={2018}
}
```
