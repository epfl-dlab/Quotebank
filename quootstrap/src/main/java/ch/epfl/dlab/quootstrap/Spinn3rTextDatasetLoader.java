package ch.epfl.dlab.quootstrap;

import java.io.Serializable;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;


public class Spinn3rTextDatasetLoader implements DatasetLoader, Serializable {

	private static final long serialVersionUID = 1432174520856283513L;
	
	private final boolean caseFold;
	private transient Configuration config = null; 
	
	public Spinn3rTextDatasetLoader()  {
		caseFold = ConfigManager.getInstance().isCaseFoldingEnabled();
	}
	
	@Override
	public JavaRDD<DatasetLoader.Article> loadArticles(JavaSparkContext sc, String datasetPath, Set<String> languageFilter) {
		if (config == null) {
			// Copy configuration
			config = new Configuration(sc.hadoopConfiguration());
			config.set("textinputformat.record.delimiter", "\n\n");
		}
		
		JavaRDD<DatasetLoader.Article> records = sc.newAPIHadoopFile(datasetPath, TextInputFormat.class, LongWritable.class, Text.class, config)
			.map(x -> new Spinn3rDocument(x._2.toString()))
			.filter(x -> !x.isGarbled
					&& x.content != null && !x.content.isEmpty()
					&& x.date != null
					&& x.urlString != null
					//&& languageFilter.contains(x.getProbableLanguage())
					)
			.map(x -> {
				final String id = x.docId;
				final String domain = x.urlString;
			    final String time = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(x.date);
			    final String version = x.version.name();
			    final String title = x.title;
			    
			    String content = x.content;
			    if (caseFold) {
			    	content = content.toLowerCase(Locale.ROOT);
			    }
			    
			    return new Article(id, content, domain, time, version, title);
			});
			//.mapToPair(x -> new Tuple2<>(x.getArticleUID(), x))
			//.reduceByKey((x, y) -> x.getArticleUID() < y.getArticleUID() ? x : y) // Deterministic distinct
			//.map(x -> x._2);
		
		/*System.out.println(sc.newAPIHadoopFile(datasetPath, TextInputFormat.class, LongWritable.class, Text.class, config)
			.map(x -> new Spinn3rDocument(x._2.toString()))
			.take(2));

		System.out.println(records.take(1));
		System.exit(0);*/
		
		//String suffix = ConfigManager.getInstance().getLangSuffix();
		//records = Utils.loadCache(records, "documents-" + suffix);
		return records;
	}
	
	public static void main(String[] args) {
		final SparkConf conf = new SparkConf()
				.setAppName("QuotationExtraction")
				.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
				.registerKryoClasses(new Class<?>[] { ArrayList.class, Token.class, Token.Type.class, Sentence.class, Pattern.class,
					Trie.class, Trie.Node.class, String[].class, Object[].class, HashMap.class, Hashed.class });
		
		if (ConfigManager.getInstance().isLocalModeEnabled()) {
			conf.setMaster("local[*]")
				.set("spark.executor.memory", "8g")
				.set("spark.driver.memory", "8g");
		}
		
		try (JavaSparkContext sc = new JavaSparkContext(conf)) {
			Spinn3rTextDatasetLoader loader = new Spinn3rTextDatasetLoader();
			
			Set<String> langFilter = new HashSet<>(ConfigManager.getInstance().getLangFilter());
			loader.loadArticles(sc, "C:\\Users\\Dario\\Documents\\EPFL\\part-r-00000.snappy", langFilter);
		}
		
		
	}
	
	public static class Article implements DatasetLoader.Article, Serializable {

		private static final long serialVersionUID = -5411421564171041258L;
		
		private final String articleUID;
		private final String articleContent;
		private final String website;
		private final String date;
		private final String version;
		private final String title;
		
		public Article(String articleUID, String articleContent, String website, String date, String version, String title) {
			this.articleUID = articleUID;
			this.articleContent = articleContent;
			this.website = website;
			this.date = date;
			this.version = version;
			this.title = title;
		}
		
		public Article(String articleUID, String articleContent, String website, String date) {
			this(articleUID, articleContent, website, date, "", "");
		}
		
		@Override
		public String getArticleUID() {
			return articleUID;
		}

		@Override
		public List<String> getArticleContent() {
			// Construct on the fly
			return new ArrayList<>(Arrays.asList(articleContent.split(" ")));
		}

		@Override
		public String getWebsite() {
			return website;
		}

		@Override
		public String getDate() {
			return date;
		}
		
		@Override
		public String getVersion() {
			return version;
		}
		
		@Override
		public String getTitle() {
			return title;
		}

		@Override
		public String toString() {
			return articleUID + ": " + articleContent;
		}
		
	}

}
