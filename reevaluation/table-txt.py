import pandas as pd

# Path to the CSV file with the averaged metrics
csv_file = '/mnt/data/medical_benchmark/reevaluation-aggregated-metrics/average_over_datasets.csv'

# Read the CSV file into a DataFrame
df_final = pd.read_csv(csv_file)

# Ensure 'resolution' is a string for consistent comparison
df_final['resolution'] = df_final['resolution'].astype(str)

# List of models
models = df_final['model'].unique()

# Metrics to include
metrics_to_include = ['AUC', 'BALACC', 'Co']

# Resolutions for first row and second row
res_first_row = ['28', '64']
res_second_row = ['128', '224']

# Output file path
output_filename = '/mnt/data/medical_benchmark/reevaluation-aggregated-metrics/metrics_output.txt'

# Open the txt file for writing
with open(output_filename, 'w') as f:
    for model in models:
        # Filter data for this model
        model_data = df_final[df_final['model'] == model]
        # Prepare first row
        first_row_values = [model]
        for metric in metrics_to_include:
            for res in res_first_row:
                # Filter data for this resolution
                row = model_data[model_data['resolution'] == res]
                if not row.empty:
                    mean_value = row[f'{metric}_mean'].values[0]
                    std_value = row[f'{metric}_std'].values[0]
                    formatted_value = '{:.4f} \\pm {:.4f}'.format(mean_value, std_value)
                else:
                    formatted_value = 'NA'
                first_row_values.append(formatted_value)
        first_row = ' & '.join(first_row_values) + ' \\\\'
        # Prepare second row
        second_row_values = [model]
        for metric in metrics_to_include:
            for res in res_second_row:
                # Filter data for this resolution
                row = model_data[model_data['resolution'] == res]
                if not row.empty:
                    mean_value = row[f'{metric}_mean'].values[0]
                    std_value = row[f'{metric}_std'].values[0]
                    formatted_value = '{:.4f} \\pm {:.4f}'.format(mean_value, std_value)
                else:
                    formatted_value = 'NA'
                second_row_values.append(formatted_value)
        second_row = ' & '.join(second_row_values) + ' \\\\'
        # Write the rows to the file
        f.write(first_row + '\n')
        f.write(second_row + '\n')

print(f"Metrics have been saved to '{output_filename}'.")
