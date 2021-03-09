spark-submit --jars spinn3r-client-3.4.05-edit.jar,stanford-corenlp-3.8.0.jar,jsoup-1.10.3.jar,guava-14.0.1.jar \
	--num-executors 25 \
	--executor-cores 16 \
	--driver-memory 128g \
	--executor-memory 128g \
	--conf "spark.executor.memoryOverhead=32768" \
	--class ch.epfl.dlab.quootstrap.QuotationExtraction \
	--master yarn \
	quootstrap.jar