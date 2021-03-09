package ch.epfl.dlab.quootstrap;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;

import ch.epfl.dlab.spinn3r.Tokenizer;
import ch.epfl.dlab.spinn3r.TokenizerImpl;

public class ExporterContext {

	private final JavaRDD<Sentence> sentences;
	private final int numPartition;

	public ExporterContext(JavaRDD<Sentence> sentences) {
		this.sentences = sentences;
		this.numPartition = ConfigManager.getInstance().getNumPartition();
	}

	public void exportResults(String exportPath, JavaSparkContext sc) throws IOException {
		sentences.repartition(numPartition).map(x -> {
			JsonObject o = new JsonObject();
			o.addProperty("articleUID", x.getArticleUid());
			o.addProperty("articleOffset", x.getIndex());

			String leftContext = "";
			String rightContext = "";
			String quotation = "";
			List<Token> tokens = x.getTokens();
			Tokenizer tokenizer = new TokenizerImpl();

			for (int i = 0; i < x.getTokenCount(); i++) {
				Token t = tokens.get(i);
				if (t.getType() == Token.Type.QUOTATION) {
					leftContext = tokenizer.untokenize(Token.getStrings(tokens.subList(0, i)));
					quotation = t.toString();
					rightContext = tokenizer.untokenize(Token.getStrings(tokens.subList(i + 1, x.getTokenCount())));
					break;
				}
			}

			o.addProperty("leftContext", leftContext);
			o.addProperty("rightContext", rightContext);
			o.addProperty("quotation", quotation);
			o.addProperty("quotationOffset", x.getQuotationOffset());
			o.addProperty("leftOffset", x.getLeftOffset());
			o.addProperty("rightOffset", x.getRightOffset());

			return new GsonBuilder().disableHtmlEscaping().create().toJson(o);
		}).filter(x -> x != null).saveAsTextFile(exportPath, GzipCodec.class);
	}
}
