import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error,mean_squared_error
# Load the Excel file
file_path = 'Strength_Data.xlsx'
excel_data = pd.ExcelFile(file_path)


# Function to process each sheet
def process_sheet(sheet_name):
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    # Get unique labels
    unique_labels = data['Label'].unique()

    # Initialize list to store processed data
    processed_data = []

    # Process data for each label
    for label in unique_labels:
        label_data = data[data['Label'] == label]

        if 'Small' in sheet_name:
            num = 2
        else:
            num = 5
        # Ensure there are enough points to form groups
        if len(label_data) < num:
            continue

        # Sort by Real values
        label_data = label_data.sort_values(by='Real').reset_index(drop=True)

        # Create groups of 5 points each
        num_groups = len(label_data) // num

        # If there are leftover points, add them to the last group
        if len(label_data) % num != 0:
            num_groups += 1

        grouped_data_list = []
        for i in range(num_groups):
            group = label_data.iloc[i * num:(i + 1) * num]
            if group.empty:
                continue
            true_mean = group['Real'].mean()
            predicted_mean = group['predicted'].mean()
            variance = mean_squared_error(group['Real'] , group['predicted'])
            stddev = root_mean_squared_error(group['Real'] , group['predicted'])
            grouped_data_list.append({
                'Real_mean': true_mean,
                'Predicted_mean': predicted_mean,
                'Variance': variance,
                'StdDev': stddev,
                'Label': label
            })

        grouped_data = pd.DataFrame(grouped_data_list)
        processed_data.append(grouped_data)

    return pd.concat(processed_data, ignore_index=True) if processed_data else pd.DataFrame()


# Process each sheet
processed_sheets = {sheet_name: process_sheet(sheet_name) for sheet_name in excel_data.sheet_names}

# Save the results to a new Excel file
output_file_path = 'processed_strength_data.xlsx'
with pd.ExcelWriter(output_file_path) as writer:
    for sheet_name, result_df in processed_sheets.items():
        if not result_df.empty:
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Processed data saved to {output_file_path}")
