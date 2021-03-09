package ch.epfl.dlab.quootstrap;

import java.io.IOException;
import java.util.stream.Collectors;

import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;

import scala.Tuple2;

public class ExporterSpeakers {

	private final JavaPairRDD<String, Iterable<SpeakerAlias>> allSpeakers;
	private final int numPartition;

	public ExporterSpeakers(JavaPairRDD<String, Iterable<SpeakerAlias>> allSpeakers) {
		this.allSpeakers = allSpeakers;
		this.numPartition = ConfigManager.getInstance().getNumPartition();
	}

	public void exportResults(String exportPath, JavaSparkContext sc) throws IOException {
		allSpeakers.repartition(numPartition).map(x -> {
			JsonObject o = new JsonObject();
			o.addProperty("articleUID", x._1);
			JsonArray names = new JsonArray();
			for (SpeakerAlias alias: x._2) {
				JsonObject current = new JsonObject();
				JsonArray ids = new JsonArray();
				JsonArray offsets = new JsonArray();
				String current_alias = alias.getAlias().stream().collect(Collectors.joining(" "));
				alias.getIds().stream().map(y -> new Tuple2<>(new JsonPrimitive(y._1), new JsonPrimitive(y._2))).forEach(y -> {
					JsonArray current_ids = new JsonArray();
					current_ids.add(y._1);
					current_ids.add(y._2);
					ids.add(current_ids);
				});
				alias.getOffsets().stream().map(y -> new Tuple2<>(new JsonPrimitive(y._1), new JsonPrimitive(y._2))).forEach(y -> {
					JsonArray current_offset = new JsonArray();
					current_offset.add(y._1);
					current_offset.add(y._2);
					offsets.add(current_offset);
				});
				current.addProperty("name", current_alias);
				current.add("ids", ids);
				current.add("offsets", offsets);
				names.add(current);
			}
			o.add("names", names);

			return new GsonBuilder().disableHtmlEscaping().create().toJson(o);
		}).filter(x -> x != null).saveAsTextFile(exportPath, GzipCodec.class);
	}

}
