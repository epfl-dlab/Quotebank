package ch.epfl.dlab.quootstrap;

import java.net.MalformedURLException;
import java.net.URL;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class Spinn3rDocument {
  private static final String dateFormat = "yyyy-MM-dd HH:mm:ss";
  public String docId = null;
  public URL url = null;
  public String urlString = null;
  public Date date = null;
  public String title = null;
  public String title_raw = null;
  public String content = null;
  public String content_raw = null;
  public List<Link> links = new ArrayList<Link>();
  public List<Quote> quotes = new ArrayList<Quote>();
  public List<Lang> langs = new ArrayList<Lang>();
  public Spinn3rVersion version = null;
  public boolean isGarbled;
  public double nonGarbageFraction;

  private boolean keepNonstandardLines;
  public List<String> nonstandardLines;

  public enum Spinn3rVersion {
    A, B, C, D, E;
  }

  public enum ContentType {
    WEB, TWITTER, FACEBOOK;

    public String toString() {
      switch (this) {
      case WEB:
        return "W";
      case TWITTER:
        return "T";
      case FACEBOOK:
        return "F";
      default:
        throw new IllegalArgumentException();
      }
    }
  }

  private static String escapeNewLines(String in) {
    return in.replaceAll("\n", "&#10;");
  }

  private static String escapeNewLinesAndTabs(String in) {
    return in.replaceAll("\n", "&#10;").replace("\t", "&#9;");
  }

  /*
   * Prints the document in the "full5" multi-line format. We also add languages and the information
   * about garbage text.
   */
  @Override
  public String toString() {
    StringBuffer str = new StringBuffer();
    if (docId != null) {
      str.append("I\t").append(docId).append("\n");
    } else {
      throw new IllegalArgumentException("Document has no docId");
    }
    if (version != null) {
      str.append("V\t").append(version).append("\n");
    } else {
      throw new IllegalArgumentException("Document has no version");
    }
    for (Lang l : langs) {
      str.append("S\t").append(l.toString()).append("\n");
    }
    str.append("G\t").append(isGarbled + "\t").append(nonGarbageFraction + "\t").append("\n");
    if (urlString != null) {
      str.append("U\t").append(escapeNewLines(urlString)).append("\n");
    } else {
      throw new IllegalArgumentException("Document has no URL");
    }
    if (date != null) {
      str.append("D\t").append(escapeNewLines(new SimpleDateFormat(dateFormat).format(date))).append("\n");
    } else {
      throw new IllegalArgumentException("Document has no date!");
    }
    if (title != null) {
      str.append("T\t").append(escapeNewLines(title)).append("\n");
    }
    if (title_raw != null) {
      str.append("F\t").append(escapeNewLines(title_raw)).append("\n");
    }
    if (content != null) {
      str.append("C\t").append(escapeNewLines(content)).append("\n");
    }
    if (content_raw != null) {
      str.append("H\t").append(escapeNewLines(content_raw)).append("\n");
    }
    for (Link l : links) {
      str.append("L\t").append(l.toString()).append("\n");
    }
    for (Quote q : quotes) {
      str.append("Q\t").append(q.toString()).append("\n");
    }
    if (keepNonstandardLines) {
      for (String s : nonstandardLines) {
        str.append(s).append("\n");
      }
    }
    return str.toString();
  }

  /*
   * Prints the document in the "F5" single-line format.
   */
  public String toStringF5() {
    StringBuffer str = new StringBuffer();
    if (urlString != null) {
      str.append("U:").append(escapeNewLinesAndTabs(urlString)).append("\t");
    } else {
      throw new IllegalArgumentException("Document has no URL");
    }
    if (date != null) {
      str.append("D:").append(escapeNewLinesAndTabs(new SimpleDateFormat(dateFormat).format(date))).append("\t");
    } else {
      throw new IllegalArgumentException("Document has no date!");
    }
    if (title != null) {
      str.append("T:").append(escapeNewLinesAndTabs(title)).append("\t");
    }
    if (title_raw != null) {
      str.append("F:").append(escapeNewLinesAndTabs(title_raw)).append("\t");
    }
    if (content != null) {
      str.append("C:").append(escapeNewLinesAndTabs(content)).append("\t");
    }
    if (content_raw != null) {
      str.append("H:").append(escapeNewLinesAndTabs(content_raw)).append("\t");
    }
    for (Link l : links) {
      str.append("L:").append(l.toString()).append("\t");
    }
    for (Quote q : quotes) {
      str.append("Q:").append(q.toString()).append("\t");
    }
    if (keepNonstandardLines) {
      for (String s : nonstandardLines) {
        str.append(s.charAt(0)).append(':').append(s.substring(2)).append("\t");
      }
    }
    return str.toString();
  }

  public void parseLine(String line) {
    char type = line.charAt(0);
    String value = line.substring(2);

    switch (type) {
    case 'I': // DocId
      this.docId = value;
      break;
    case 'V': // Version
      this.version = Spinn3rVersion.valueOf(value);
      break;
    case 'S': // Languages
      String[] split = value.split("\t", 2);
      String lng = split[0];
      double prob = Double.valueOf(split[1]);
      this.langs.add(new Lang(lng, prob));
      break;
    case 'G': // Garbled info
      String[] split1 = value.split("\t", 2);
      this.isGarbled = Boolean.valueOf(split1[0]);
      this.nonGarbageFraction = Double.valueOf(split1[1]);
      break;
    case 'U': // Url
      urlString = value;
      try {
        this.url = new URL(value);
      } catch (MalformedURLException e) {
      }
      break;
    case 'D': // Date
      try {
        this.date = new SimpleDateFormat(dateFormat).parse(value);
      } catch (ParseException e) {
        e.printStackTrace();
        System.exit(-1);
      }
      break;
    case 'T': // Title
      this.title = value;
      break;
    case 'F': // Title raw
      this.title_raw = value;
      break;
    case 'C': // Content
      this.content = value;
      break;
    case 'H': // Content raw
      this.content_raw = value;
      break;
    case 'L': // Links
      String[] split2 = value.split("\t", 3);
      int startPos = Integer.valueOf(split2[0]);
      if (split2[1].equals("")) {
        this.links.add(new Link(startPos, split2[2]));
      } else {
        int length = Integer.valueOf(split2[1]);
        this.links.add(new Link(startPos, length, split2[2]));
      }
      break;
    case 'Q': // Quotes
      String[] split3 = value.split("\t", 3);
      int startPos1 = Integer.valueOf(split3[0]);
      int length = Integer.valueOf(split3[1]);
      this.quotes.add(new Quote(startPos1, length, split3[2]));
      break;
    default: // Unknown value
      throw new IllegalArgumentException("Illegal type character '" + type
          + "' found during parsing spinn3r document.");
    }
  }

  /*
   * Construct the Spinn3rDocument object from multi-line string. Used for parsing the documents
   * stored in Hadoop.
   */
  public Spinn3rDocument(String doc, boolean keepNonstandardLines) {
    this.keepNonstandardLines = keepNonstandardLines;
    if (keepNonstandardLines) {
      nonstandardLines = new ArrayList<String>();
    }
    // we use String.indexOf() since it is faster than String.split() method
    // append one final newline since the code below needs it
    doc = doc + '\n';
    int pos = 0, end;
    while ((end = doc.indexOf('\n', pos)) >= 0) {
      String line = doc.substring(pos, end);
      try {
        parseLine(line);
      } catch (IllegalArgumentException e) {
        if (keepNonstandardLines && line.charAt(0) >= 'A' && line.charAt(0) <= 'Z'
            && line.charAt(1) == '\t') {
          // If we want to keep nonstandard lines, store this one (if it's of the required form
          // <LETTER><TAB>...).
          nonstandardLines.add(line);
        } else {
          // Otherwise ignore it.
        }
      }
      pos = end + 1;
    }
  }

  public Spinn3rDocument(String doc) {
    this(doc, true);
  }

  /*
   * Append a language to this document.
   */
  public void appendLang(String lang, double prob) {
    this.langs.add(new Lang(lang, prob));
  }

  /*
   * Class for storing links about this record.
   * 
   * A value of -1 for startPos/length means that the link appears in the title, not in the content.
   */
  public static class Link {
    public int startPos;
    // For the older versions (up to, and incl., full5), this is empty, i.e., null.
    public Integer length;
    public String url;

    public Link(int startPos, String url) {
      this.startPos = startPos;
      this.length = null;
      this.url = url;
    }

    public Link(int startPos, int length, String url) {
      this.startPos = startPos;
      this.length = length;
      this.url = url;
    }

    @Override
    public String toString() {
      return String.format("%d\t%s\t%s", startPos, length == null ? "" : length,
          escapeNewLinesAndTabs(url));
    }
  }

  /*
   * Class for storing quotes about this record.
   * 
   * A value of -1 for startPos/length means that the quote appears in the title, not in the
   * content.
   */
  public static class Quote {
    //
    public int startPos;
    public int length;
    public String text;

    public Quote(int startPos, int length, String text) {
      this.startPos = startPos;
      this.length = length;
      this.text = text;
    }

    @Override
    public String toString() {
      return String.format("%s\t%d\t%s", startPos, length, escapeNewLinesAndTabs(text));
    }
  }

  /*
   * Class for storing languages about this record
   */
  public class Lang {
    public String lang;
    public double prob;

    public Lang(String lang, double prob) {
      this.lang = lang;
      this.prob = prob;
    }

    @Override
    public String toString() {
      return String.format("%s\t%f", lang, prob);
    }
  }

  /*
   * Returns true if document has language whose probability >= 0.8.
   */
  public boolean hasProbableLanguage() {
    if (this.langs.size() > 0 && this.langs.get(0).prob >= 0.8) {
      return true;
    }
    return false;
  }

  /*
   * Returns the first language if it is probable. Else it returns null.
   */
  public String getProbableLanguage() {
    if (this.hasProbableLanguage()) {
      return this.langs.get(0).lang;
    }
    return null;
  }
}
