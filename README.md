Project Structure
The project is organized into a series of modules, each focusing on a distinct step in the information retrieval pipeline:

Data Preprocessing:

Tokenization: Breaks down raw text into individual words or tokens.
Normalization: Converts all tokens to lowercase and removes punctuation, ensuring uniformity across terms.
Stopword Removal: Filters out common words (like "and", "the") that carry little meaning to improve relevance.
Stemming/Lemmatization: Reduces tokens to their root form, helping consolidate terms with similar meanings.
Term-Document Matrix:

This matrix represents the frequency of each term in each document, forming a basis for further retrieval algorithms.
Inverted Index:

The inverted index maps each unique term to the list of documents it appears in, along with its frequency. This index enables efficient lookups by term, supporting quick retrieval of documents related to search queries.
TF-IDF Calculation:

Term Frequency (TF): Measures the occurrence of each term in a document.
Inverse Document Frequency (IDF): Evaluates the significance of terms based on their rarity across the entire collection.
TF-IDF: Combines TF and IDF scores to weigh terms that are unique and relevant, refining search results by emphasizing important keywords in each document.
