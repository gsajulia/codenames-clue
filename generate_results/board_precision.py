import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

    colors = ['#8338ec', '#52b69a', '#3a86ff', '#fb6f92']
    # Bar chart
    plt.figure(figsize=(15, 8))
    for i, metric in enumerate(metrics):
        metric_values = [sum(all_model_metrics[model][metric]) / len(all_model_metrics[model][metric]) for model in model_names]
        plt.bar([pos + i * width for pos in x], metric_values, width=width, label=metric, color=colors[i])

    plt.xticks([pos + 1.5 * width for pos in x], model_names)
    plt.ylabel('Percentage')
    plt.title('Combined Metrics for All Models')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'combined_model_metrics.png'))
    plt.close()

    # Scatter plot between Accuracy (avg of Accuracy 1 and Accuracy 2) and Good Clue Rate for each model
    plt.figure(figsize=(15, 8))
    
    accuracy_values = []
    good_clue_values = []
    jitter_strength = 0.5
    
    for model in model_names:
        accuracy_avg = (sum(all_model_metrics[model]['Accuracy 1']) / len(all_model_metrics[model]['Accuracy 1']) +
                        sum(all_model_metrics[model]['Accuracy 2']) / len(all_model_metrics[model]['Accuracy 2'])) / 2
        avg_good_clue_rate = sum(all_model_metrics[model]['Good Clue Rate']) / len(all_model_metrics[model]['Good Clue Rate'])
        
        accuracy_jitter = accuracy_avg + np.random.uniform(-jitter_strength, jitter_strength)
        good_clue_jitter = avg_good_clue_rate + np.random.uniform(-jitter_strength, jitter_strength)

        accuracy_values.append(accuracy_jitter)
        good_clue_values.append(good_clue_jitter)

        plt.scatter(accuracy_jitter, good_clue_jitter, s=100, label=model, color=colors[model_names.index(model) % len(colors)])


    # Adjust scale for the graph
    plt.xlim(min(accuracy_values) - 5, max(accuracy_values) + 5)  # Added small margin
    plt.ylim(min(good_clue_values) - 5, max(good_clue_values) + 5)  # Added small margin
    
    plt.xlabel('Average Accuracy (%)')
    plt.ylabel('Good Clue Rate (%)')
    plt.title('Scatter Plot: Accuracy vs Good Clue Rate for All Models')
    plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'scatter_accuracy_vs_good_clue.png'))
    plt.close()

input_folder = 'board_precision'
output_folder = 'graphs'
all_dfs = read_all_files_from_folder(input_folder)
plot_combined_metrics(all_dfs, output_folder)