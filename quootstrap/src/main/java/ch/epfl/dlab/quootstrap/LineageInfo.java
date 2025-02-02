package ch.epfl.dlab.quootstrap;

import java.io.Serializable;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Instances of this class contain the lineage information associated with a quotation-speaker pair.
 *
 */
public final class LineageInfo implements Serializable {

	private static final long serialVersionUID = -6380483748689471512L;
	
	private final List<Pattern> patterns;
	private final List<Sentence> sentences;
	private final List<List<String>> aliases;
	private final double confidence;
	
	public LineageInfo(List<Pattern> patterns, List<Sentence> sentences, List<List<String>> aliases, double confidence) {
		this.patterns = patterns;
		this.sentences = sentences;
		this.confidence = confidence;
		this.aliases = aliases;
	}
	
	public List<Pattern> getPatterns() {
		return patterns;
	}
	
	public List<Sentence> getSentences() {
		return sentences;
	}
	
	public List<List<String>> getAliases() {
		return aliases;
	}
	
	public double getConfidence() {
		return confidence;
	}
	
	@Override
	public String toString() {
		return "{Confidence: " + confidence + ", Patterns: " + patterns + ", Sentences: "
			+ sentences.stream().map(x -> "<" + x.getKey().toString() + "> " + x.toString()).collect(Collectors.toList()) + "}";
	}
}
