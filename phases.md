# Spinn3r character encoding: description of Phases A through E

Quotebank was extracted from news articles that had been collected from the media aggregation service Spinn3r (now called [Datastreamer](https://www.datastreamer.io)).
The Spinn3r data was collected over the course of over a decade.
During this time, the client-side code used for collecting the data changed several times, and various character-encoding-related issues led to different representations of the original text at different times. Most issues relate to capital letters and non-ASCII characters.

This document, although not entirely conclusive, is the result of an "archæological" endeavor to reconstruct the history of the various character encodings used at different times during Spinn3r data collection.
Based on our insights, we divide the 12 years spanned by the Spinn3r corpus into five phases (Phases A through E), detailed below.
Non-ASCII characters are most relevant for non-English text; capitalization, however, matters for English as well.
For most users, the key takeaways of this document are these:

1. Text was lowercased in Phases A, B, and C, whereas the original capitalization was maintained in Phases D and E.
2. Non-ASCII characters are properly represented only in Phase E.

(This document is based on an initial write-up made on 6 June 2014.)


## Phase A (until 2010-07-13)

Spinn3r's probably UTF-8-encoded data was read as Latin-1 (a.k.a. ISO-8859-1). UTF-8 has potentially several bytes per character, while Latin-1 has always one byte per character. That is, a single character from the original data now looks like two characters. For instance, Unicode code point U+00E4 ("Latin small letter a with diaeresis", a.k.a. "ä") is represented by the two-byte code C3A4 in UTF-8. Reading the bytes C3A4 as Latin-1 results in the two-character sequence "Ã¤", since C3 encodes "Ã" in Latin-1, and A4, "¤".
Then, **lowercasing** was performed on the garbled text, making it even more garbled. For instance, "Ã¤" became "ã¤".
Finally, the data was written to disk as UTF-8.

**Approximate solution:**
Take the debugging table from [http://www.i18nqa.com/debug/utf8-debug.html](https://web.archive.org/web/20210228174408/http://www.i18nqa.com/debug/utf8-debug.html), look for the garbled and lower-cased sequences and replace them by their original character.
Note that the garbling is not bijective, but since most of the garbled sequences are highly unlikely (e.g., "ã¤"), this should be mostly fine.


## Phase B (2010-07-14 to 2010-07-26)

For just about two weeks, the data seems to have been read as UTF-8 and written as Latin-1 (i.e., the other way round than in phase A).
Non-Latin-1 characters are printed as "?". However, there also seem to be a very few cases as in Phase A.
All text was **lowercased** in this phase.

**Approximate solution:**
Simply read the data as Latin-1.


## Phase C (2010-07-27 to 2013-04-28)

The data was written to disk as ASCII, such that all non-ASCII characters (including Latin-1 characters) appear as "?".
All text was **lowercased** in this phase.

**Approximate solution:**
None. We simply need to byte (haha...) the bullet and deal with the question marks.


## Phase D (2013-04-29 to 2014-05-21)

Attempt 1 at fixing the above legacy issues:
capitalization is kept as in the original text obtained from Spinn3r.
However, due to a bad BASH environment variable, text was written to disk as ASCII, such that non-ASCII characters still appear as "?".


## Phase E (since 2014-05-22)

Attempt 2 at fixing the above legacy issues:
capitalization is kept as in the original text obtained from Spinn3r, and output is now finally written as proper UTF-8 Unicode.
