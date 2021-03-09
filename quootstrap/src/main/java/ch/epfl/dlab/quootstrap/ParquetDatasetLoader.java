package ch.epfl.dlab.quootstrap;

import java.util.Set;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import scala.Tuple4;

public class ParquetDatasetLoader implements DatasetLoader {

	private static SparkSession session;
	
	private static void initialize(JavaSparkContext sc) {
		if (session == null) {
			session = new SparkSession(sc.sc());
		}
	}
	
	@Override
	public JavaRDD<Article> loadArticles(JavaSparkContext sc, String datasetPath, Set<String> languageFilter) {
		ParquetDatasetLoader.initialize(sc);
		Dataset<Row> df = ParquetDatasetLoader.session.read().parquet(datasetPath);
		
		/* Expected schema:
		 * articleUID: String
		 * website: String
		 * content: String
		 * date: String
		 */
		
		return df.select(df.col("articleUID"), df.col("website"), df.col("content"), df.col("date"))
			.map(x -> {
					String uid = x.getString(0);
					String url = x.getString(1);
					String content = x.getString(2);
					String date = x.getString(3);
					return new Tuple4<>(uid, content, url, date);
				}, Encoders.tuple(Encoders.STRING(), Encoders.STRING(), Encoders.STRING(), Encoders.STRING()))
			.javaRDD()
			.map(x -> new Spinn3rTextDatasetLoader.Article(x._1(), x._2(), x._3(), x._4()));
	}
}
