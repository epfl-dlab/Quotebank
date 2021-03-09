package ch.epfl.dlab.quootstrap;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import scala.Tuple2;

public class SpeakerAlias implements Serializable {

	private static final long serialVersionUID = -6037497222642880562L;
	
	private final List<String> aliasTokens;
	private final List<Tuple2<String, String>> ids;
	private final List<Tuple2<Integer, Integer>> offsets;
	
	public SpeakerAlias(List<String> aliasTokens, List<Tuple2<String, String>> ids, Tuple2<Integer, Integer> offset) {
		this.aliasTokens = aliasTokens;
		this.ids = ids;
		this.offsets = new ArrayList<>();
		this.offsets.add(offset);
	}
	
	public List<String> getAlias() {
		return aliasTokens;
	}
	
	public List<Tuple2<String, String>> getIds() {
		return ids;
	}
	
	public List<Tuple2<Integer, Integer>> getOffsets() {
		return offsets;
	}
	
	public void addOffset(Tuple2<Integer, Integer> offset) {
		offsets.add(offset);
	}

	@Override
	public int hashCode() {
		return aliasTokens.hashCode() ^ ids.hashCode();
	}
	
	@Override
	public boolean equals(Object o) {
		if (o instanceof SpeakerAlias) {
			SpeakerAlias oa = (SpeakerAlias) o;
			return oa.ids.equals(ids); //oa.aliasTokens.equals(aliasTokens) &&
		}
		return false;
	}
	
}
