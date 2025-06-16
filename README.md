# Code for Data of climate risk public viewsand government response indices for China
## Installation Dependencies
To install the required dependencies, run the following command:
```bash

pip install -r requirements.txt

```
## Data
We use the Opinion and Response Dataset, which is collected from the Message board for Leaders between January 2011 and December 2023.
## Index Calculation
### Expression of Public Views Index
The `calculate_expressions_of_public_views_index.py` script is used to calculate the concern index for each message. This script processes the dataset to compute the concern index, which reflects the level of public concern in the messages.
### Government Response Index
- **Response Status and Timeliness:**
The `calculate_response_status_time_index.py` script calculates the response status (whether a response was given) and the timeliness of the response (the time difference between the message and the response).
- **Response Quality:**
The `response_quality_index.py` script calculates the quality of the response by comparing the content of the message and the response using word embeddings and other natural language processing techniques.
## Result
The results from the above calculations are aggregated at the city level using the `aggregate_public_views_index_government_response_index.py` script. The example code provided demonstrates the aggregation on a monthly basis. Readers can modify the script to perform daily or yearly aggregations as needed.

## Note
According to the requirements of the journal Scientific Data, we have uploaded an identical copy of the code to Figshare. The difference is that the code on Figshare includes a pre-trained word2vec file based on the full dataset of messages from the Message Board for Leaders, which can be directly used to generate the climate risk expressions of public views and government responses indices. The corresponding URL for Figshare is: https://doi.org/10.6084/m9.figshare.28120904.

## Cite
Sun, X., Li, L., Shen, Y., Sheng, Y., Zhang, D., & Ji, Q. (2025). Climate risk expressions of public views and government responses in China. Scientific Data, 12(1), 1-10.
