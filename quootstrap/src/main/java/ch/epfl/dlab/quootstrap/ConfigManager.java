package ch.epfl.dlab.quootstrap;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.stream.Collectors;

public final class ConfigManager {

	private static final String CONFIG_FILE = "resources/config.properties";
	
	private static final ConfigManager INSTANCE = new ConfigManager();
	
	private final String datasetPath;
	private final String namesPath;
	private final int numIterations;
	private final int minQuotationSize;
	private final int maxQuotationSize;
	private final boolean caseSensitive;
	private final List<String> langFilter;
	
	private final String newsDatasetLoader;
	
	private final boolean exportEnabled;
	private final String exportPath;
	private final boolean doQuoteAttribution;
	
	private final boolean contextExportEnabled;
	private final String contextOutputPath;
	
	private final int numPartition;
	
	private final boolean localModeEnabled;
	
	private final double confidenceThreshold;
	private final List<Double> clusteringThresholds;
	
	private final boolean mergingEnabled;
	private final boolean doDedup;
	private final int mergingShingleSize;
	
	private final boolean cacheEnabled;
	private final String cachePath;
	
	private final String groundTruthPath;
	private final boolean finalEvaluationEnabled;
	private final boolean intermediateEvaluationEnabled;
	
	private final boolean dumpPatternsEnabled;
	private final boolean caseFoldingEnabled;
	
	private String outputPath;

	private final boolean speakersEnabled;
	private final String speakersPath;
	private final boolean articleEnabled;
	private final String articlePath;
	
	private ConfigManager() {
		final Properties prop = new Properties();
		try {
			prop.load(new FileInputStream(CONFIG_FILE));
		} catch (IOException e) {
			throw new IllegalArgumentException("Unable to read config file", e);
		}
		
		datasetPath = prop.getProperty("NEWS_DATASET_PATH");
		namesPath = prop.getProperty("PEOPLE_DATASET_PATH");
		numIterations = Integer.parseInt(
				prop.getProperty("NUM_ITERATIONS"));
		minQuotationSize = Integer.parseInt(
				prop.getProperty("MIN_QUOTATION_SIZE"));
		maxQuotationSize = Integer.parseInt(
				prop.getProperty("MAX_QUOTATION_SIZE"));
		caseSensitive = prop.getProperty("CASE_SENSITIVE").equals("true");
		langFilter = Arrays.asList(prop.getProperty("LANGUAGE_FILTER").split("\\|"));
		
		newsDatasetLoader = prop.getProperty("NEWS_DATASET_LOADER");
		
		doQuoteAttribution = prop.getProperty("DO_QUOTE_ATTRIBUTION").equals("true");

		exportEnabled = prop.getProperty("EXPORT_RESULTS").equals("true");
		exportPath = prop.getProperty("EXPORT_PATH");
		
		contextExportEnabled = prop.getProperty("EXPORT_CONTEXT").equals("true");
		contextOutputPath = prop.getProperty("CONTEXT_PATH");
		
		speakersEnabled = prop.getProperty("EXPORT_SPEAKERS").equals("true");
		speakersPath = prop.getProperty("SPEAKERS_PATH");
		
		articleEnabled = prop.getProperty("EXPORT_ARTICLE").equals("true");
		articlePath = prop.getProperty("ARTICLE_PATH");
		
		numPartition = Integer.parseInt(
				prop.getProperty("NUM_PARTITIONS"));
		
		localModeEnabled = prop.getProperty("LOCAL_MODE").equals("true");
		
		confidenceThreshold = Double.parseDouble(
				prop.getProperty("PATTERN_CONFIDENCE_THRESHOLD"));
		clusteringThresholds = Arrays.asList(prop.getProperty("PATTERN_CLUSTERING_THRESHOLDS").split("\\|"))
				.stream()
				.map(Double::parseDouble)
				.collect(Collectors.toList());
		
		mergingEnabled = prop.getProperty("ENABLE_QUOTATION_MERGING").equals("true");
		doDedup = prop.getProperty("ENABLE_DEDUPLICATION").equals("true");
		mergingShingleSize = Integer.parseInt(
				prop.getProperty("MERGING_SHINGLE_SIZE"));
		
		cacheEnabled = prop.getProperty("ENABLE_CACHE").equals("true");
		cachePath = prop.getProperty("CACHE_PATH");
		
		groundTruthPath = prop.getProperty("GROUND_TRUTH_PATH");
		finalEvaluationEnabled = prop.getProperty("ENABLE_FINAL_EVALUATION").equals("true");
		intermediateEvaluationEnabled = prop.getProperty("ENABLE_INTERMEDIATE_EVALUATION").equals("true");
		
		dumpPatternsEnabled = prop.getProperty("DEBUG_DUMP_PATTERNS").equals("true");
		caseFoldingEnabled = prop.getProperty("DEBUG_CASE_FOLDING").equals("true");
		
		outputPath = "";
	}

	public static ConfigManager getInstance() {
		return INSTANCE;
	}

	public String getDatasetPath() {
		return datasetPath;
	}

	public String getNamesPath() {
		return namesPath;
	}

	public int getNumIterations() {
		return numIterations;
	}
	
	public boolean isCaseSensitive() {
		return caseSensitive;
	}

	public List<String> getLangFilter() {
		return langFilter;
	}
	
	public String getLangSuffix() {
		return langFilter.stream().sorted().collect(Collectors.joining("-"));
	}

	public boolean isLocalModeEnabled() {
		return localModeEnabled;
	}

	public double getConfidenceThreshold() {
		return confidenceThreshold;
	}

	public List<Double> getClusteringThresholds() {
		return clusteringThresholds;
	}

	public boolean isCacheEnabled() {
		return cacheEnabled;
	}

	public String getCachePath() {
		return cachePath;
	}

	public String getGroundTruthPath() {
		return groundTruthPath;
	}

	public boolean isFinalEvaluationEnabled() {
		return finalEvaluationEnabled;
	}

	public boolean isIntermediateEvaluationEnabled() {
		return intermediateEvaluationEnabled;
	}

	public boolean isDumpPatternsEnabled() {
		return dumpPatternsEnabled;
	}
	
	public boolean isCaseFoldingEnabled() {
		return caseFoldingEnabled;
	}

	public boolean isMergingEnabled() {
		return mergingEnabled;
	}

	public int getMergingShingleSize() {
		return mergingShingleSize;
	}

	public String getOutputPath() {
		return outputPath;
	}

	public void setOutputPath(String outputPath) {
		this.outputPath = outputPath;
	}

	public boolean isExportEnabled() {
		return exportEnabled;
	}

	public String getExportPath() {
		return exportPath;
	}
	
	public boolean isSpeakersEnabled() {
		return speakersEnabled;
	}

	public String getSpeakersPath() {
		return speakersPath;
	}
	
	public boolean isArticleEnabled() {
		return articleEnabled;
	}

	public String getArticlePath() {
		return articlePath;
	}

	public String getNewsDatasetLoader() {
		return newsDatasetLoader;
	}

	public boolean isContextExportEnabled() {
		return contextExportEnabled;
	}

	public String getContextOutputPath() {
		return contextOutputPath;
	}
	
	public int getNumPartition() {
		return numPartition;
	}

	public boolean isDoQuoteAttribution() {
		return doQuoteAttribution;
	}

	public boolean isDoDedupEnabled() {
		return doDedup;
	}

	public int getMinQuotationSize() {
		return minQuotationSize;
	}

	public int getMaxQuotationSize() {
		return maxQuotationSize;
	}
}
