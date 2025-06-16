import pandas as pd
import jieba
import pkuseg  # https://github.com/lancopku/pkuseg-python
from concurrent.futures import ProcessPoolExecutor

# Initialize the segmenter
seg = pkuseg.pkuseg()

# Load the text dataset
text_data = pd.read_csv("Message and Response Dataset.csv")

# Load the dictionary dataset
risk_words = set(pd.read_excel("climate loss concern dictionary.xlsx")['risk'])
positive_words = set(pd.read_excel("climate loss concern dictionary.xlsx")['positive'])
negative_words = set(pd.read_excel("climate loss concern dictionary.xlsx")['negative'])

# Function to calculate the concern index for a given content
def calculate_concern_index(content):
    # Segment the content into words
    words = seg.cut(content)
    # Calculate the total number of words
    total_words = len(words)
    
    # Count the number of risk words
    RW = sum(1 for word in words if word in risk_words)
    
    # Count the number of negative words
    NW = sum(1 for word in words if word in negative_words)
    
    # Count the number of positive words
    PW = sum(1 for word in words if word in positive_words)
    
    # Avoid division by zero
    if NW + PW == 0:
        return 0, 0, 0, 0
    
    # Calculate the concern index
    concern = 100 * (RW / total_words) * ((((NW - PW) / (NW + PW)) + 1) / 2)
    
    return concern

# Apply the function to each row in the 'Opinion' column and add the result to the DataFrame
text_data['concern_index'] = text_data['Opinion'].apply(calculate_concern_index)

# Export the new DataFrame with the concern index to a CSV file
text_data.to_csv("Message and Response Dataset_concern.csv", index=False)

print("Concern index calculation completed and exported to Message and Response Dataset_concern.csv")