package ch.epfl.dlab.quootstrap;

import java.io.Serializable;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import com.google.gson.Gson;

import ch.epfl.dlab.spinn3r.EntryWrapper;

public class Spinn3rDatasetLoader implements DatasetLoader, Serializable {

	private static final long serialVersionUID = 7689367526740335445L;
	
	private final boolean caseFold;
	
	public Spinn3rDatasetLoader() {
		caseFold = ConfigManager.getInstance().isCaseFoldingEnabled();
	}
	
	@Override
	public JavaRDD<DatasetLoader.Article> loadArticles(JavaSparkContext sc, String datasetPath, Set<String> languageFilter) {
		return sc.textFile(datasetPath)
			.map(x -> new Gson().fromJson(x, EntryWrapper.class).getPermalinkEntry())
			//.filter(x -> languageFilter.contains(x.getLanguage()))
			.map(x -> {
				String domain;
				try {
					final URL url = new URL(x.getUrl());
					domain = url.getHost();
				} catch (MalformedURLException e) {
					domain = "";
				}
			    
			    String time;
			    if (!x.getLastPublished().isEmpty() && !x.getDateFound().isEmpty()) {
			    	time = Collections.min(Arrays.asList(x.getLastPublished(), x.getDateFound()));
			    } else if (!x.getLastPublished().isEmpty()) {
			    	time = x.getLastPublished();
			    } else {
			    	time = x.getDateFound();
			    }
			    
			    List<String> tokenizedContent = x.getContent();
			    if (caseFold) {
			    	tokenizedContent.replaceAll(token -> token.toLowerCase(Locale.ROOT));
			    }
			    
			    return new Article(Long.parseLong(x.getIdentifier()), tokenizedContent, domain, time);
			});
	}
	
	public static class Article implements DatasetLoader.Article {
		
		private final long articleUID;
		private final List<String> articleContent;
		private final String website;
		private final String date;
		
		public Article(long articleUID, List<String> articleContent, String website, String date) {
			this.articleUID = articleUID;
			this.articleContent = articleContent;
			this.website = website;
			this.date = date;
		}
		
		@Override
		public String getArticleUID() {
			return Long.toString(articleUID);
		}

		@Override
		public List<String> getArticleContent() {
			return articleContent;
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
			return null;
		}

		@Override
		public String getTitle() {
			return null;
		}
		
		
		
	}

}
