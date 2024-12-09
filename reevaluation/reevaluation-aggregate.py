import os
import pandas as pd
import glob

# Define the directory containing the per-dataset CSV files
directory = '<directory>'

# Gets list of per-dataset CSV files
csv_files = glob.glob(os.path.join(directory, '*_aggregated_metrics.csv'))

dataframes = []

# Reads each per-dataset CSV file and add a 'dataset' column
for file in csv_files:
    dataset_name = os.path.basename(file).replace('_aggregated_metrics.csv', '')
    df = pd.read_csv(file)
    df['dataset'] = dataset_name
    dataframes.append(df)

# Combines all DataFrames into one
df_combined = pd.concat(dataframes, ignore_index=True)

# Melts the DataFrame to long format
id_vars = ['trainingRegimen', 'model', 'dataset']
value_vars = [col for col in df_combined.columns if col not in id_vars]

df_long = df_combined.melt(
    id_vars=id_vars,
    value_vars=value_vars,
    var_name='metric_resolution_stat',
    value_name='value'
)

# Drops rows where value is NaN
df_long = df_long.dropna(subset=['value'])

# Splits 'metric_resolution_stat' into 'metric', 'resolution', 'stat' (mean or std)
df_long[['metric', 'resolution', 'stat']] = df_long['metric_resolution_stat'].str.rsplit('_', n=2, expand=True)

# Drops the 'metric_resolution_stat' column
df_long = df_long.drop(columns='metric_resolution_stat')

# Pivots the DataFrame so that 'stat' becomes columns ('mean' and 'std')
df_pivot = df_long.pivot_table(
    index=['model', 'resolution', 'metric', 'dataset'],
    columns='stat',
    values='value'
).reset_index()

# Calculates the average of the mean and std over all datasets
df_avg = df_pivot.groupby(['model', 'resolution', 'metric']).agg({'mean': 'mean', 'std': 'mean'}).reset_index()

# Pivots to get metrics as columns
df_final = df_avg.pivot_table(
    index=['model', 'resolution'],
    columns='metric',
    values=['mean', 'std']
)

# Flattens the MultiIndex columns
df_final.columns = ['{}_{}'.format(metric, stat) for stat, metric in df_final.columns]

# Reset index to turn 'model' and 'resolution' into columns
df_final = df_final.reset_index()

# Saving the final DataFrame to CSV
output_filename = '<output_path>'
df_final.to_csv(output_filename, index=False)

print(f"Averaged metrics have been saved to '{output_filename}'.")
