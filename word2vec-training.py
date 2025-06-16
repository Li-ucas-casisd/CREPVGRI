import pandas as pd
from gensim.models import Word2Vec
import pkuseg
import logging
import os

# Configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def load_data(file_path):
    """
    Load data from Message and Response Dataset.csv file
    """
    df = pd.read_csv(file_path)
    print(f"Data size: {len(df)}")
    print(df.head())  # Display first few rows
    return df

def preprocess_text(df, text_columns=['Opinion Content', 'Response Content']):
    """
    Extract text from specified columns and perform word segmentation using pkuseg
    """
    # Load pkuseg model
    seg = pkuseg.pkuseg()  # Use default configuration
    
    # Combine text from both columns
    texts = []
    for col in text_columns:
        texts.extend(df[col].dropna().tolist())  # Remove NA values and convert to list
    
    # Word segmentation
    tokenized_texts = []
    for text in texts:
        words = seg.cut(str(text))  # Segment text using pkuseg
        tokenized_texts.append([word for word in words if word.strip() != ""])  # Remove empty strings
    
    return tokenized_texts

def train_word2vec(tokenized_texts, vector_size=200, window_size=6, min_count=6, output_file='renmin_board.200.6.bin'):
    """
    Train Word2Vec model and save with specified output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Train model
    model = Word2Vec(
        sentences=tokenized_texts,  # Tokenized texts
        vector_size=vector_size,    # Word vector dimension
        window=window_size,         # Context window size
        min_count=min_count,        # Minimum word frequency
        workers=4,                 # Number of parallel threads
        sg=1,                      # Use skip-gram (1) instead of CBOW (0)
        hs=0,                      # Use negative sampling
        negative=5                  # Number of negative samples
    )
    
    # Save model in binary format to generate all required files
    model.wv.save_word2vec_format(output_file, binary=True)
    print(f"Model saved with files: {output_file} and associated .npy files")
    
    return model

if __name__ == "__main__":
    # File path
    file_path = 'Message and Response Dataset.csv'
    
    # 1. Load data
    df = load_data(file_path)
    
    # 2. Extract text and perform word segmentation
    tokenized_texts = preprocess_text(df)
    
    # 3. Train Word2Vec model
    model = train_word2vec(
        tokenized_texts,
        vector_size=300,  # Word vector dimension
        window_size=6,   # Context window size
        min_count=6,     # Minimum word frequency
        output_file='renmin_board.200.6.bin'  # Output file name
    )
    