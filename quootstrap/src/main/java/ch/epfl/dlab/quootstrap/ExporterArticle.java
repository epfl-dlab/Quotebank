package ch.epfl.dlab.quootstrap;

import java.io.IOException;

import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;

import ch.epfl.dlab.quootstrap.DatasetLoader.Article;

public class ExporterArticle {

	private final JavaRDD<Article> allArticles;
	// private final int numPartition;

	public ExporterArticle(JavaRDD<Article> allArticles) {
		this.allArticles = allArticles;
		// this.numPartition = ConfigManager.getInstance().getNumPartition();
	}

	public void exportResults(String exportPath, JavaSparkContext sc) throws IOException {
		allArticles.map(x -> {  //repartition(numPartition).
			JsonObject o = new JsonObject();
			o.addProperty("articleUID", x.getArticleUID());
			o.addProperty("content", String.join(" ", x.getArticleContent()));
			o.addProperty("date", x.getDate());
			o.addProperty("website", x.getWebsite());
			o.addProperty("phase", x.getVersion());
			o.addProperty("title", x.getTitle());

			return new GsonBuilder().disableHtmlEscaping().create().toJson(o);
		}).filter(x -> x != null).saveAsTextFile(exportPath, GzipCodec.class);
	}

}
