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

    # Check if the new columns exist and include them if present
    combined_accuracy = (accuracy_1 + accuracy_2) / 2  # Combine both accuracies

    if 'is_correct_3' in df.columns:
        accuracy_3 = df['is_correct_3'].mean() * 100
        combined_accuracy = (accuracy_1 + accuracy_2 + accuracy_3) / 3  # Combine all three accuracies

    return combined_accuracy, correct_pairs_rate, good_clue_rate

def plot_combined_metrics(all_dfs, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_model_metrics = {}
    for i, df in enumerate(all_dfs):
        filename = f'file_{i}'
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            combined_accuracy, correct_pairs_rate, good_clue_rate = calculate_metrics(model_df)
            
            # Include is_good_words_batch metric if it exists
            if 'is_good_words_batch' in df.columns:
                is_good_words_batch_rate = df['is_good_words_batch'].mean() * 100
            else:
                is_good_words_batch_rate = None  # Set as None if the column doesn't exist

            if model not in all_model_metrics:
                all_model_metrics[model] = {
                    'Accuracy': [],
                    'Correct Pairs Rate': [],
                    'Good Clue Rate': [],
                }
                
            if is_good_words_batch_rate is not None:
                all_model_metrics[model] = {
                'Accuracy': [],
                'Correct Pairs Rate': [],
                'Good Clue Rate': [],
                'Good Words Batch Rate': [],
            }
            all_model_metrics[model]['Accuracy'].append(combined_accuracy)
            all_model_metrics[model]['Correct Pairs Rate'].append(correct_pairs_rate)
            all_model_metrics[model]['Good Clue Rate'].append(good_clue_rate)
            if is_good_words_batch_rate is not None:
                all_model_metrics[model]['Good Words Batch Rate'].append(is_good_words_batch_rate)

    model_names = sorted(list(all_model_metrics.keys()))
    metrics = ['Accuracy', 'Correct Pairs Rate', 'Good Clue Rate']
    if is_good_words_batch_rate is not None:
        metrics.append('Good Words Batch Rate')
    x = range(len(model_names))
    width = 0.2
    background_color_empty = '#edede9'
    colors = ['#8338ec', '#52b69a', '#3a86ff', '#fb6f92', '#f1c40f']
    
    # Bar chart
    plt.figure(figsize=(15, 8))
    for i, metric in enumerate(metrics):
        metric_values = []
        for model in model_names:
            # Prevent division by zero
            if len(all_model_metrics[model][metric]) > 0:
                metric_values.append(sum(all_model_metrics[model][metric]) / len(all_model_metrics[model][metric]))
            else:
                metric_values.append(0)  # If no data, append 0 or a suitable value

        plt.bar([pos + i * width for pos in x], [100] * len(x), width=width, color=background_color_empty, zorder=1)
        plt.bar([pos + i * width for pos in x], metric_values, width=width, label=metric, color=colors[i])

    plt.xticks([pos + 1.5 * width for pos in x], model_names)
    plt.ylabel('Percentage')
    plt.title('Combined Metrics for All Models')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'combined_model_metrics.png'))
    plt.close()

    # Scatter plot between Accuracy and Good Clue Rate for each model
    plt.figure(figsize=(15, 8))
    
    accuracy_values = []
    good_clue_values = []
    jitter_strength = 0.5
    
    for model in model_names:
        combined_accuracy_avg = sum(all_model_metrics[model]['Accuracy']) / len(all_model_metrics[model]['Accuracy']) if len(all_model_metrics[model]['Accuracy']) > 0 else 0
        avg_good_clue_rate = sum(all_model_metrics[model]['Good Clue Rate']) / len(all_model_metrics[model]['Good Clue Rate']) if len(all_model_metrics[model]['Good Clue Rate']) > 0 else 0
        
        accuracy_jitter = combined_accuracy_avg + np.random.uniform(-jitter_strength, jitter_strength)
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