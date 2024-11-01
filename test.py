import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from natsort import natsorted
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Folder path containing the text files
folder_path = 'C:/Users/Heba Omarr/Desktop/Basic-search-engine-main/files'

# Tokenization, stop word removal, and stemming
def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens
def or_postings(posting1, posting2):
    p1 = 0
    p2 = 0
    result = list()
    while p1 < len(posting1) and p2 < len(posting2):
        if posting1[p1] == posting2[p2]:
            result.append(posting1[p1]) #2
            p1 += 1
            p2 += 1
        elif posting1[p1] > posting2[p2]:
            result.append(posting2[p2])
            p2 += 1
        else:
            result.append(posting1[p1])
            p1 += 1
    while p1 < len(posting1):
        result.append(posting1[p1])
        p1 += 1
    while p2 < len(posting2):
        result.append(posting2[p2])
        p2 += 1
    return result
documents = []
fullfile = []
file_name = natsorted(os.listdir('files'))
for files in file_name:
    with open('files/' + files, 'r') as f:
        text = f.read()
        fullfile.append(text)
        preprocessed_text = preprocess_text(text)
        documents.append(preprocessed_text)

#print(documents)
#print('-------------------------------')
#print(fullfile)
#print('-------------------------------')
print('choose 1 for Term document matrix ')
print('choose 2 for inverted index ')
print('choose 3 for TF_IDF ')

ch = int(input("Enter your choice (1 , 2, or 3): "))

if ch == 1:
    unique_terms = {term for doc in fullfile for term in doc.split()}

    doc_term_matrix = {}

    for term in unique_terms:
        doc_term_matrix[term] = []

        for doc in fullfile:
            if term in doc:
                doc_term_matrix[term].append(1)
            else:
                doc_term_matrix[term].append(0)

    # print(doc_term_matrix)

    docs_array = np.array(fullfile, dtype='object')
    # print(docs_array)

    vfinal = None
    query = input("Enter Query: ")

    q_tokens = word_tokenize(query)
    for qword in q_tokens:
        v1 = np.array(doc_term_matrix[qword])
        if vfinal is None:
            vfinal = v1
        else:
            vfinal = np.bitwise_and(v1, vfinal)

    print('Files are:')
    cn = 1
    for i in vfinal:
        if i == 1:
            print('Document', cn)
        cn += 1
elif ch == 2:
    
    inverted_index = {}

    for i, doc in enumerate(fullfile):
        for term in doc.split():
            if term in inverted_index:
                inverted_index[term].add(i)
            else:
                inverted_index[term] = {i}

    # print(inverted_index)
    
    def and_postings(posting1, posting2):
        p1 = 0
        p2 = 0
        result = list()
        while p1 < len(posting1) and p2 < len(posting2):
            if posting1[p1] == posting2[p2]:
                result.append(posting1[p1])
                p1 += 1
                p2 += 1
            elif posting1[p1] > posting2[p2]:
                p2 += 1
            else:
                p1 += 1
        return result

    vfinal = list()
    query = input("Enter Query: ")

    q_tokens = word_tokenize(query)
    for qword in q_tokens:
        pl_1 = list(inverted_index[qword])

        if vfinal == list():
            vfinal = pl_1
        else:
            vfinal = and_postings(pl_1, vfinal)
    print('Files are:')
    for i in vfinal:
        print('Document', i)
elif ch == 3:
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(fullfile)

    query = input("Enter your query: ")

    query_tfidf = tfidf_vectorizer.transform([query])

    similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix)

    top_n = 2  
    sorted_indices = np.argsort(similarity_scores[0])[::-1][:top_n]

    print(f"Top {top_n} relevant documents for the query:")
    for index in sorted_indices:
        print(f"Document {index + 1}")
