import os
import pandas as pd
from collections import defaultdict
import numpy as np

# Define the directories, one where the model-specific csv files are stored, one where the
directory = '/mnt/data/medical_benchmark/reevaluation'
save_dir = '/mnt/data/medical_benchmark/reevaluation-aggregated-metrics'

# A dictionary to hold datasets and their corresponding files
dataset_files = defaultdict(list)

# The training regimen that was used
trainingRegimen = 'endToEnd'

# Collect all possible resolutions and metrics
all_resolutions = set()
metrics_names = ['ACC', 'AUC', 'BALACC', 'Co', 'Prec']

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('_test_metrics.csv') and trainingRegimen in filename:
        # Removes the '.csv' extension
        name = os.path.splitext(filename)[0]
        # Splits the name on '_'
        parts = name.split('_')
        if len(parts) >= 6:
            # Extract components
            dataset = parts[0]
            resolution = parts[1]
            model = '_'.join(parts[3:-3])
            seed = parts[-3][1:]  # Removing the 's' prefix
            # Append the full path and other info to the list for this dataset
            dataset_files[dataset].append({
                'filename': os.path.join(directory, filename),
                'resolution': resolution,
                'trainingRegimen': trainingRegimen,
                'model': model,
                'seed': seed
            })
            # Collect resolutions
            all_resolutions.add(resolution)
        else:
            print(f"Filename '{filename}' does not match expected format.")
            continue

all_resolutions = sorted(all_resolutions)  # For consistent ordering

for dataset, files_info in dataset_files.items():
    # Grouping the files by (trainingRegimen, model)
    file_groups = defaultdict(list)
    for info in files_info:
        key = (info['trainingRegimen'], info['model'])
        file_groups[key].append({
            'filename': info['filename'],
            'resolution': info['resolution'],
            'seed': info['seed']
        })

    # List to store results
    results = []

    for key, files_info_list in file_groups.items():
        # key is (trainingRegimen, model)
        trainingRegimen, model = key
        # For each resolution, collect metrics
        resolution_metrics = {}
        for info in files_info_list:
            resolution = info['resolution']
            file = info['filename']
            # Reads the CSV file into a DataFrame
            df = pd.read_csv(file)
            # Extracts the metrics columns
            try:
                metrics = df[metrics_names]
            except KeyError as e:
                print(f"Error reading metrics from {file}: {e}")
                continue
            if resolution not in resolution_metrics:
                resolution_metrics[resolution] = []
            resolution_metrics[resolution].append(metrics.values.flatten())

        # Initialize result dictionary (to be turned into csv later) to fill in the next steps
        result = {
            'trainingRegimen': trainingRegimen,
            'model': model,
        }

        # Calculates mean and std for each resolution
        for resolution in all_resolutions:
            if resolution in resolution_metrics:
                metrics_array = np.array(resolution_metrics[resolution])
                mean_metrics = np.mean(metrics_array, axis=0)
                std_metrics = np.std(metrics_array, axis=0)
                for i, metric_name in enumerate(metrics_names):
                    result[f'{metric_name}_{resolution}_mean'] = mean_metrics[i]
                    result[f'{metric_name}_{resolution}_std'] = std_metrics[i]
            else:
                # If the resolution is not present for this model, fill with NaN
                for metric_name in metrics_names:
                    result[f'{metric_name}_{resolution}_mean'] = np.nan
                    result[f'{metric_name}_{resolution}_std'] = np.nan

        results.append(result)

    # Converts results to DataFrame
    results_df = pd.DataFrame(results)

    # Fills missing values with NaN
    results_df = results_df.fillna(np.nan)

    # Saves the DataFrame to CSV without the index
    output_filename = f'{save_dir}/{dataset}_aggregated_metrics.csv'
    results_df.to_csv(output_filename, index=False)
    print(f"Aggregated metrics for dataset '{dataset}' have been saved to '{output_filename}'.")
