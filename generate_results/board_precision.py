import os
import pandas as pd
import matplotlib.pyplot as plt

def read_all_files_from_folder(folder_path, file_extension='.csv'):
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(file_extension):
            file_path = os.path.join(folder_path, file_name)
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
            except Exception as e:
                print(f"Error reading the file {file_name}: {e}")
    return all_data

def calculate_metrics(df):
    accuracy_1 = df['is_correct_1'].mean() * 100
    accuracy_2 = df['is_correct_2'].mean() * 100
    correct_pairs_rate = df['correct_pair'].mean() * 100
    good_clue_rate = df['is_good_clue'].mean() * 100
    return accuracy_1, accuracy_2, correct_pairs_rate, good_clue_rate

def plot_combined_metrics(all_dfs, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_model_metrics = {}
    for i, df in enumerate(all_dfs):
        filename = f'file_{i}'
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            accuracy_1, accuracy_2, correct_pairs_rate, good_clue_rate = calculate_metrics(model_df)
            if model not in all_model_metrics:
                all_model_metrics[model] = {
                    'Accuracy 1': [],
                    'Accuracy 2': [],
                    'Correct Pairs Rate': [],
                    'Good Clue Rate': [],
                }
            all_model_metrics[model]['Accuracy 1'].append(accuracy_1)
            all_model_metrics[model]['Accuracy 2'].append(accuracy_2)
            all_model_metrics[model]['Correct Pairs Rate'].append(correct_pairs_rate)
            all_model_metrics[model]['Good Clue Rate'].append(good_clue_rate)

    model_names = list(all_model_metrics.keys())
    metrics = ['Accuracy 1', 'Accuracy 2', 'Correct Pairs Rate', 'Good Clue Rate']
    x = range(len(model_names))
    width = 0.2

    plt.figure(figsize=(15, 8))
    for i, metric in enumerate(metrics):
        metric_values = [sum(all_model_metrics[model][metric]) / len(all_model_metrics[model][metric]) for model in model_names]
        plt.bar([pos + i * width for pos in x], metric_values, width=width, label=metric)

    plt.xticks([pos + 1.5 * width for pos in x], model_names)
    plt.ylabel('Percentage')
    plt.title('Combined Metrics for All Models')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'combined_model_metrics.png'))
    plt.close()

input_folder = 'board_precision'
output_folder = 'graphs'
all_dfs = read_all_files_from_folder(input_folder)
plot_combined_metrics(all_dfs, output_folder)