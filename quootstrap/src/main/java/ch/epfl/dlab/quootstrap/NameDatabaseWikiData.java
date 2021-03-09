package ch.epfl.dlab.quootstrap;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import ch.epfl.dlab.spinn3r.Tokenizer;
import ch.epfl.dlab.spinn3r.TokenizerImpl;
import scala.Tuple2;

public class NameDatabaseWikiData implements Serializable {

	private static final long serialVersionUID = 4372945856241816022L;
	
	private final HashTrie trie;
	private transient JavaPairRDD<List<String>, List<Tuple2<String, String>>> peopleRDD; // Maps alias to its (WikiData ID, standard name) list
	
	public NameDatabaseWikiData(JavaSparkContext sc, String knowledgeFile) {
		trie = new HashTrie(getNamesRDD(sc, knowledgeFile).collect(), ConfigManager.getInstance().isCaseSensitive());
	}
	
	private JavaRDD<Tuple2<List<String>, List<Tuple2<String, String>>>> getNamesRDD(JavaSparkContext sc, String fileName) {
		peopleRDD = sc.textFile(fileName)
				.mapToPair(x -> {
					String[] chunks = x.split("\t", -1);
					// System.out.println(chunks);
					List<String> aliases = new ArrayList<>(Arrays.asList(chunks[2].split("\\|")));
					List<String> normalized = new ArrayList<>(Arrays.asList(chunks[5].split("\\|")));
					
					aliases.add(0, chunks[1]); // Add standard name to alias
					
					return new Tuple2<>(new Tuple2<>(chunks[0], chunks[1]), new Tuple2<>(aliases, normalized)); // ((id, standard name), (aliases, spellings))
				})
				.filter(x -> x._2._2.size() > 0) // Avoid empty names
				.flatMapValues(x -> {
					// Rebuild alias list
					
					Set<String> normalizedSet = new HashSet<>(x._2);
					
					Tokenizer tokenizer = new TokenizerImpl();
					List<List<String>> newAliases = new ArrayList<>();
					
					for (String name : x._1) {
						String normalized = StaticRules.normalizePre(name);
						List<String> normTokenized = tokenizer.tokenize(normalized);
						List<String> tokenized = tokenizer.tokenize(name);
						
						String normReconstruction = String.join(" ", normTokenized);
						String reconstruction = String.join(" ", tokenized);
						
						if (normalizedSet.contains(normReconstruction)) {
							newAliases.add(tokenized);
							
							if (!normReconstruction.equalsIgnoreCase(reconstruction)) {
								newAliases.add(normTokenized);
							}
						}
					}
					return new ArrayList<>(newAliases);
				}) // 
				.filter(x -> x._2.size() > 1) // Number of tokens
				.mapToPair(Tuple2::swap) // (alias), (id, standard name) can be multiple per id, standard name
				.groupByKey()
				.mapValues(x -> {
					// Put ambiguous alias' id together
					List<Tuple2<String, String>> ids = new ArrayList<>();
					x.forEach(y -> ids.add(y));
					return ids;
				});
				//.reduceByKey((x, y) -> x._1.compareTo(y._1) == -1 ? x : y) // Deduplicate deterministically
		
		peopleRDD = Utils.loadCache(peopleRDD, "people-database");
		
		return peopleRDD.map(x -> x);
	}
	
	public JavaPairRDD<List<String>, List<Tuple2<String, String>>> getPeopleRDD() {
		return peopleRDD;
	}
	
	public HashTriePatternMatcher newMatcher() {
		return new HashTriePatternMatcher(trie);
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
			NameDatabaseWikiData db = new NameDatabaseWikiData(sc, "C:\\Users\\Dario\\Documents\\Università\\Semester project 2\\wikidata_people_ALIVE_FILTERED-NAMES.tsv");
			Utils.dumpRDDLocal(db.getPeopleRDD().map(x -> x.toString()), "debug_wikidata.txt");
		}
		
		
	}
}
