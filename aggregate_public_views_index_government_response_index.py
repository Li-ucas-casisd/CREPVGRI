import pandas as pd

# Read the three CSV files
df_concern = pd.read_csv('Message and Response Dataset_concern.csv')
df_status_time = pd.read_csv('Message and Response Dataset_status_time.csv')
df_quality = pd.read_csv('Message and Response Dataset_quality.csv')

# Function to convert date column to the start of the month
def convert_to_month_start(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    df[date_column] = df[date_column].dt.to_period('M').dt.start_time
    return df

# Convert the 'Message time' column to the start of the month for each DataFrame
df_concern = convert_to_month_start(df_concern, 'Message time')
df_status_time = convert_to_month_start(df_status_time, 'Message time')
df_quality = convert_to_month_start(df_quality, 'Message time')

# Merge the DataFrames
df_merged = df_concern.merge(df_status_time, on=['Message time', 'City'], how='left')
df_merged = df_merged.merge(df_quality, on=['Message time', 'City'], how='left')

# Group by 'Message time' and 'City' and aggregate the data
grouped = df_merged.groupby(['Message time', 'City']).agg(
    concern_index_sum=('concern_index', 'sum'),
    response_status_mean=('response_status', 'mean'),
    response_timeliness_mean=('response_timeliness', 'mean'),
    quality_mean=('quality', 'mean')
).reset_index()

# Function to normalize values
def normalize(x, min_val, max_val, reverse=False):
    if reverse:
        return (max_val - x) / (max_val - min_val)
    else:
        return (x - min_val) / (max_val - min_val)

# Find the maximum and minimum values for each metric
max_response_status = grouped['response_status_mean'].max()
min_response_status = grouped['response_status_mean'].min()
max_response_timeliness = grouped['response_timeliness_mean'].max()
min_response_timeliness = grouped['response_timeliness_mean'].min()
max_quality = grouped['quality_mean'].max()
min_quality = grouped['quality_mean'].min()

# Apply normalization
grouped['normalized_response_status'] = grouped['response_status_mean'].apply(lambda x: normalize(x, min_response_status, max_response_status))
grouped['normalized_response_timeliness'] = grouped['response_timeliness_mean'].apply(lambda x: normalize(x, min_response_timeliness, max_response_timeliness, reverse=True))
grouped['normalized_quality'] = grouped['quality_mean'].apply(lambda x: normalize(x, min_quality, max_quality))

# Calculate the response index
grouped['response_index'] = grouped[['normalized_response_status', 'normalized_response_timeliness', 'normalized_quality']].mean(axis=1)

# Save the aggregated results to a new CSV file
output_file = 'Aggregated_Dataset_by_Month_and_City.csv'
grouped.to_csv(output_file, index=False, encoding="utf-8")

print(f"Aggregated results have been successfully exported to {output_file}")
