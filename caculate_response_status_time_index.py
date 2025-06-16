import pandas as pd
from datetime import datetime
from dateutil.parser import parse

# Read the CSV file
df = pd.read_csv('Message and Response Dataset.csv')

# Ensure the date columns are in string format
df['Message time'] = df['Message time'].astype(str)
df['Response time'] = df['Response time'].astype(str)

# Add a "response_status" column
df['response_status'] = df['Response'].notnull().astype(int)

# Define a function to calculate the difference in days between two dates
def calculate_days_difference(start_time, end_time):
    if pd.isnull(start_time) or pd.isnull(end_time):
        return None
    try:
        start = parse(start_time)
        end = parse(end_time)
        return (end - start).days
    except ValueError:
        # If parsing fails, catch the exception and print a message
        print(f"Unable to parse time: {start_time} or {end_time}")
        return None

# Add a "response_timeliness" column
df['response_timeliness'] = df.apply(lambda row: calculate_days_difference(row['Message time'], row['Response time']), axis=1)

# Save the processed DataFrame to a new CSV file
df.to_csv('Message and Response Dataset_status_time.csv', index=False)

print("Processing completed and exported to Message and Response Dataset_status_time.csv")
