package ch.epfl.dlab.quootstrap;

import java.util.List;
import java.util.Set;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

/**
 * This interface is used for loading a news dataset.
 * The handling of the format is up to the concrete implementation.
 */
public interface DatasetLoader {

	/**
	 * Load all articles as a Java RDD.
	 * @param sc the JavaSparkContext used for loading the RDD
	 * @param datasetPath the path of the dataset.
	 * @param languageFilter a set that contains the requested language codes.
	 * @return a JavaRDD of Articles
	 */
	JavaRDD<Article> loadArticles(JavaSparkContext sc, String datasetPath, Set<String> languageFilter);
	
	/**
	 * A news article, identified by a unique long identifier.
	 */
	public interface Article {
		
		/** Get the unique identifier of this article. */
		String getArticleUID();
		
		/** Get the content of this article in tokenized format. */
		List<String> getArticleContent();
		
		/** Get the domain name in which this article was found. */
		String getWebsite();
		
		/** Get the date of this article. */
		String getDate();
		
		/** Get the version of the article */
		String getVersion();
		
		/** Get the title of this article */
		String getTitle();
		
	}
}
