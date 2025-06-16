import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
from gensim.models import KeyedVectors
import pkuseg  # https://github.com/lancopku/pkuseg-python
from concurrent.futures import ProcessPoolExecutor

# Load the dataset
df = pd.read_excel("Message and Response Dataset.csv")

# Load the stopwords
with open("baidu_stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = set(f.read().splitlines())

# Initialize the segmenter
seg = pkuseg.pkuseg()  # Load the model with default configuration

# Tokenize the Opinion and Response columns and remove stopwords
df["Tokenized_Opinion"] = df["Opinion"].apply(
    lambda x: [
        str(token)
        for token in seg.cut(str(x))
        if token.strip() != "" and token not in stopwords
    ]
)
df["Tokenized_Response"] = df["Response"].apply(
    lambda x: [
        str(token)
        for token in seg.cut(str(x))
        if token.strip() != "" and token not in stopwords
    ]
)

# Build the dictionary and TF-IDF model
docs = df["Tokenized_Opinion"].tolist() + df["Tokenized_Response"].tolist()
DICTIONARY = Dictionary(docs)
docs_bow = [DICTIONARY.doc2bow(doc) for doc in docs]
TFIDF_MODEL = TfidfModel(docs_bow)

# Convert tokenized content to bag-of-words representation
df["Opinion_bow"] = df["Tokenized_Opinion"].apply(lambda x: DICTIONARY.doc2bow(x))
df["Response_bow"] = df["Tokenized_Response"].apply(lambda x: DICTIONARY.doc2bow(x))

# Load the pre-trained word2vec model
w2v_model = KeyedVectors.load("renmin_board.200.6.bin")
termsim_index = WordEmbeddingSimilarityIndex(w2v_model.wv)
termsim_matrix = SparseTermSimilarityMatrix(termsim_index, DICTIONARY, TFIDF_MODEL)

# Quality calculation function
def calculate_quality(row, termsim_matrix):
    question = row["Opinion_bow"]
    answer = row["Response_bow"]
    quality = termsim_matrix.inner_product(question, answer, normalized=(True, True))
    return quality

# Parallel processing function
def parallel_apply(df, func, termsim_matrix, workers=4):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda row: func(row, termsim_matrix), [row for _, row in df.iterrows()]))
    return results

# Calculate qualities in parallel
df["quality"] = parallel_apply(df, calculate_quality, termsim_matrix)

# Set quality to None if the response is empty
df['quality'] = df.apply(lambda row: None if pd.isna(row['Response']) else row['quality'], axis=1)

# Export the DataFrame to a CSV file
df.to_csv("Message and Response Dataset_quality.csv", index=False)

print("Quality calculation completed and exported to Message and Response Dataset_quality.csv")