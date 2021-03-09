package ch.epfl.dlab.quootstrap;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;

import scala.Tuple2;

public class HashTrie implements Serializable {

	private static final long serialVersionUID = -9164247688163451454L;
	
	private final Node rootNode;
	private final boolean caseSensitive;
	
	public HashTrie(Iterable<Tuple2<List<String>, List<Tuple2<String, String>>>> substrings, boolean caseSensitive) {
		this.rootNode = new Node(null, caseSensitive, Collections.emptyList());
		this.caseSensitive = caseSensitive;
		substrings.forEach(this::insertSubstring);
	}
	
	private void insertSubstring(Tuple2<List<String>, List<Tuple2<String, String>>> substring) {
		Node current = rootNode;
		
		for (int i = 0; i < substring._1.size(); i++) {
			String token = substring._1.get(i);
			String key = caseSensitive ? token : token.toLowerCase(Locale.ROOT);
			
			Node next = current.findChild(key);
			if (next == null) {
				next = new Node(token, caseSensitive, Collections.emptyList());
				current.children.put(key, next);
			}
			current = next;
		}

		current.terminal = true;
		current.addIds(substring._2);
	}
	
	public List<List<String>> getAllSubstrings() {
		List<List<String>> allSubstrings = new ArrayList<>();
		
		List<String> currentSubstring = new ArrayList<>();
		DFS(rootNode, currentSubstring, allSubstrings);
		return allSubstrings;
	}
	
	public Node getRootNode() {
		return rootNode;
	}
	
	private void DFS(Node current, List<String> currentSubstring, List<List<String>> allSubstrings) {
		if (current.isTerminal()) {
			allSubstrings.add(new ArrayList<>(currentSubstring));
		}
		
		for (Map.Entry<String, Node> next : current) {
			currentSubstring.add(next.getKey());
			DFS(next.getValue(), currentSubstring, allSubstrings);
			currentSubstring.remove(currentSubstring.size() - 1);
		}
	}
	
	public static class Node implements Iterable<Map.Entry<String, Node>>, Serializable {
		
		private static final long serialVersionUID = -4344489198225825075L;
		
		private final Map<String, Node> children;
		private final boolean caseSensitive;
		private final String value;
		private boolean terminal;
		private List<Tuple2<String, String>> ids; // (id, standard name)
		
		public Node(String value, boolean caseSensitive, Collection<Tuple2<String, String>> ids) {
			this.children = new HashMap<>();
			this.caseSensitive = caseSensitive;
			this.value = value;
			this.terminal = false;
			this.ids = new ArrayList<>(ids);
		}
		
		public void addIds(Collection<Tuple2<String, String>> ids) {
			this.ids.addAll(ids);
		}
		
		public boolean hasChildren() {
			return !children.isEmpty();
		}
		
		public Node findChild(String token) {
			return children.get(caseSensitive ? token : token.toLowerCase(Locale.ROOT));
		}
		
		public boolean isTerminal() {
			return terminal;
		}
		
		public String getValue() {
			return value;
		}
		
		public List<Tuple2<String, String>> getIds() {
			return ids;
		}

		@Override
		public Iterator<Entry<String, Node>> iterator() {
			return Collections.unmodifiableMap(children).entrySet().iterator();
		}
	}
}
