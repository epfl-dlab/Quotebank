package ch.epfl.dlab.quootstrap;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple4;

public class Exporter {

	private final JavaPairRDD<String, String> quotationMap;
	
	/** Map article UID to a tuple (full quotation, website, date) */
	private final JavaPairRDD<Tuple2<String, Integer>, Tuple3<String, String, String>> articles;

	private final int numPartition;
	
	public Exporter(JavaSparkContext sc, JavaRDD<Sentence> sentences, NameDatabaseWikiData people)  {
		
		final Set<String> langSet = new HashSet<>(ConfigManager.getInstance().getLangFilter());
		this.articles = QuotationExtraction.getConcreteDatasetLoader().loadArticles(sc,
				ConfigManager.getInstance().getDatasetPath(), langSet)
			.mapToPair(x -> new Tuple2<>(new Tuple2<>(x.getWebsite(), x.getDate()), new Tuple2<>(x.getArticleContent(), x.getArticleUID())))
			.flatMapValues(x -> ContextExtractor.extractQuotations(x._1(), x._2()))
			.mapToPair(x -> new Tuple2<>(x._2.getKey(), new Tuple3<>(x._2.getQuotation(), x._1._1, x._1._2)));
		
		this.quotationMap = computeQuotationMap(sc);
		this.numPartition = ConfigManager.getInstance().getNumPartition();
	}
	
	private JavaPairRDD<String, String> computeQuotationMap(JavaSparkContext sc) {
		Set<String> langSet = new HashSet<>(ConfigManager.getInstance().getLangFilter());
		
		// Reconstruct quotations (from the lower-case canonical form to the full form)
		return QuotationExtraction.getConcreteDatasetLoader().loadArticles(sc,
				ConfigManager.getInstance().getDatasetPath(), langSet)
			.flatMap(x -> ContextExtractor.extractQuotations(x.getArticleContent(), x.getArticleUID()).iterator())
			.mapToPair(x -> new Tuple2<>(StaticRules.canonicalizeQuotation(x.getQuotation()), x.getQuotation()))
			.reduceByKey((x, y) -> {
				// Out of multiple possibilities, get the longest quotation
				if (x.length() > y.length()) {
					return x;
				} else if (x.length() < y.length()) {
					return y;
				} else {
					// Lexicographical comparison to ensure determinism
					return x.compareTo(y) == -1 ? x : y;
				}
			});
	}
	
	public void exportResults(JavaPairRDD<String, Tuple2<List<Tuple2<String, String>>, LineageInfo>> pairs) {
		String exportPath = ConfigManager.getInstance().getExportPath();
		
		JavaPairRDD<String, Tuple4<Tuple2<String, Integer>, String, String, String>> articleMap = pairs.mapToPair(x -> new Tuple2<>(x._1, x._2._2)) // (canonical quotation, lineage info)
			.flatMapValues(x -> {
				// (key)
				List<Tuple2<String, Integer>> values = new ArrayList<>();
				for (int i = 0; i < x.getPatterns().size(); i++) {
					values.add(x.getSentences().get(i).getKey());
				}
				return values;
			}) // (canonical quotation, key)
			.mapToPair(Tuple2::swap) // (key, canonical quotation)
			.join(this.articles) // (key, (canonical quotation, (website, date)))
			.mapToPair(x -> new Tuple2<>(x._2._1, new Tuple4<>(x._1, x._2._2._1(), x._2._2._2(), x._2._2._3()))); // (canonical quotation, (key, full quotation, website, date))
		
		pairs // (canonical quotation, (speaker, lineage info))
			.join(quotationMap)
			.mapValues(x -> new Tuple3<>(x._1._1, x._1._2, x._2)) // (canonical quotation, (speakers, lineage info, full quotation))
			.cogroup(articleMap)
			.map(t -> {
				
				String canonicalQuotation = t._1;
				Map<Tuple2<String, Integer>, Tuple3<String, String, String>> articles = new HashMap<>();
				t._2._2.forEach(x -> {
					articles.put(x._1(), new Tuple3<>(x._2(), x._3(), x._4())); // (key, (full quotation, website, date))
				});
				
				Iterator<Tuple3<List<Tuple2<String, String>>, LineageInfo, String>> it = t._2._1.iterator();
				if (!it.hasNext()) {
					return null;
				}
				Tuple3<List<Tuple2<String, String>>, LineageInfo, String> data = it.next();
				
				if (data._2().getPatterns().size() != articles.size()) {
					return null;
				}
				
				JsonArray ids = new JsonArray();
				data._1().stream().map(y -> new JsonPrimitive(y._1)).forEach(y -> ids.add(y));
				
				JsonArray names = new JsonArray();
				data._1().stream().map(y -> new JsonPrimitive(y._2)).forEach(y -> names.add(y));

				
				JsonObject o = new JsonObject();
				o.addProperty("quotation", data._3());
				o.addProperty("canonicalQuotation", canonicalQuotation);
				o.addProperty("numSpeakers", data._1().size());
				o.add("speaker", names);
				o.add("speakerID", ids);
				o.addProperty("confidence", data._2().getConfidence()); // Tuple confidence
				o.addProperty("numOccurrences", data._2().getPatterns().size());
				
				JsonArray occurrences = new JsonArray();
				for (int i = 0; i < data._2().getPatterns().size(); i++) {
					JsonObject occ = new JsonObject();
					Tuple2<String, Integer> key = data._2().getSentences().get(i).getKey();
					occ.addProperty("articleUID", key._1);
					occ.addProperty("articleOffset", data._2().getSentences().get(i).getIndex());
					occ.addProperty("extractedBy", data._2().getPatterns().get(i).toString(false));
					occ.addProperty("patternConfidence", data._2().getPatterns().get(i).getConfidenceMetric());
					occ.addProperty("quotation", articles.get(key)._1());
					
					
					String matchedTokens = String.join(" ", data._2().getAliases().get(i));
					occ.addProperty("matchedSpeakerTokens", matchedTokens);
					occ.addProperty("website", articles.get(key)._2());
					String date = articles.get(key)._3();
					if (!date.isEmpty()) {
						occ.addProperty("date", date);
					}
					occurrences.add(occ);
				}
				o.add("occurrences", occurrences);

				return new GsonBuilder().disableHtmlEscaping().create().toJson(o);
			})
			.filter(x -> x != null)
			.repartition(numPartition)
			.saveAsTextFile(exportPath, GzipCodec.class);
	}
	
}
