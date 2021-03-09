package ch.epfl.dlab.quootstrap;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import scala.Tuple2;
import scala.Tuple3;

public class HashTriePatternMatcher {

	private final HashTrie trie;
	private final List<String> currentMatch;
	private List<Tuple2<String, String>> currentIds;

	public HashTriePatternMatcher(HashTrie trie) {
		this.trie = trie;
		currentMatch = new ArrayList<>();
	}

	public boolean match(List<Token> tokens) {
		List<Tuple3<Integer, List<String>, List<Tuple2<String, String>>>> longestMatches = new ArrayList<>();
		boolean result = false;
		for (int i = 0; i < tokens.size(); i++) {
			currentMatch.clear();
			if (matchImpl(tokens, trie.getRootNode(), i)) {
				result = true;
				longestMatches.add(new Tuple3<>(i, new ArrayList<>(currentMatch), new ArrayList<>(currentIds)));
			}
		}

		if (result) {
			// Get longest match out of multiple matches
			int maxLen = Collections.max(longestMatches, (x, y) -> Integer.compare(x._2().size(), y._2().size()))._2()
					.size();
			longestMatches.removeIf(x -> x._2().size() != maxLen);

			currentMatch.clear();
			currentMatch.addAll(longestMatches.get(0)._2());
			currentIds = longestMatches.get(0)._3();
		}

		return result;
	}

	public List<SpeakerAlias> multiMatch(List<Token> tokens) {
		ArrayList<SpeakerAlias> matches = new ArrayList<>();
		for (int i = 0; i < tokens.size(); i++) {
			currentMatch.clear();
			if (matchImpl(tokens, trie.getRootNode(), i)) {
				Tuple2<Integer, Integer> offset = new Tuple2<>(i, i + currentMatch.size());
				SpeakerAlias speaker = new SpeakerAlias(new ArrayList<>(currentMatch), new ArrayList<>(currentIds),
						offset);
				int idx = matches.indexOf(speaker);
				if (idx >= 0) {
					matches.get(idx).addOffset(offset);
				} else {
					matches.add(speaker);
				}
				i += currentMatch.size() - 1; // Avoid matching subsequences
			}
		}

		return new ArrayList<>(matches);
	}

	public boolean match(Sentence s) {
		List<Token> tokens = s.getTokens();
		List<Tuple3<Integer, List<String>, List<Tuple2<String, String>>>> longestMatches = new ArrayList<>();
		boolean result = false;
		for (int i = 0; i < tokens.size(); i++) {
			currentMatch.clear();
			if (matchImpl(tokens, trie.getRootNode(), i)) {
				result = true;
				longestMatches.add(new Tuple3<>(i, new ArrayList<>(currentMatch), new ArrayList<>(currentIds)));
			}
		}

		if (result) {
			int maxLen = Collections.max(longestMatches, (x, y) -> Integer.compare(x._2().size(), y._2().size()))._2()
					.size();
			longestMatches.removeIf(x -> x._2().size() != maxLen);

			// If there are multiple speakers with max length, select the one that is
			// nearest to the quotation
			currentMatch.clear();
			if (longestMatches.size() > 1) {
				int quotationIdx = -1;
				for (int i = 0; i < tokens.size(); i++) {
					if (tokens.get(i).getType() == Token.Type.QUOTATION) {
						quotationIdx = i;
						break;
					}
				}
				final int qi = quotationIdx;
				Tuple3<Integer, List<String>, List<Tuple2<String, String>>> nearest = Collections.min(longestMatches,
						(x, y) -> {
							int delta1 = Math.abs(x._1() - qi);
							int delta2 = Math.abs(y._1() - qi);
							return Integer.compare(delta1, delta2);
						});
				currentMatch.addAll(nearest._2());
				currentIds = nearest._3();
			} else {
				currentMatch.addAll(longestMatches.get(0)._2());
				currentIds = longestMatches.get(0)._3();
			}
		}

		return result;
	}

	public SpeakerAlias getLongestMatch() {
		return new SpeakerAlias(new ArrayList<>(currentMatch), new ArrayList<>(currentIds), new Tuple2<>(0, 0));
	}

	private boolean matchImpl(List<Token> tokens, HashTrie.Node current, int i) {

		if (i == tokens.size()) {
			return false;
		}

		if (tokens.get(i).getType() != Token.Type.GENERIC) {
			return false;
		}

		String tokenStr = tokens.get(i).toString();
		HashTrie.Node next = current.findChild(tokenStr);
		if (next != null) {
			currentMatch.add(tokenStr); // next.getValue());
			boolean result = false;
			if (next.isTerminal()) {
				// Match found
				currentIds = next.getIds();
				result = true;
			}

			// Even if a match is found, try to match a longer sequence
			if (matchImpl(tokens, next, i + 1) || result) {
				return true;
			}

			currentMatch.remove(currentMatch.size() - 1);
		}

		return false;
	}
}
