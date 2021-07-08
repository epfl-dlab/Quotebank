spark-submit \
    --master yarn \
    --num-executors 16 \
    --executor-cores 32 \
    --driver-memory 16g \
    --executor-memory 64g \
    --py-files pygtrie.py \
    --conf spark.pyspark.python=python3 \
    --conf spark.driver.maxResultSize=0 \
    --conf spark.sql.shuffle.partitions=2048 \
    --conf spark.executor.memoryOverhead=16g \
    --conf spark.blacklist.enabled=true \
    --conf spark.reducer.maxReqsInFlight=10 \
    --conf spark.shuffle.io.retryWait=10s \
    --conf spark.shuffle.io.maxRetries=10 \
    --conf spark.shuffle.io.backLog=2048 \
    "$@"
