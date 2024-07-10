from model_training_and_evaluation_utils import pre_trained_embeddings_faiss_eval, two_towers_embeddings_eval
from db_utilities import read_table
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

"""
This script is only for call and run the needed steps in order to evaluate and compare
the two main explored approeaches: simple retrieval with embeddings from the pre-trained model
and retrieval with the refined version of the embeddings

The functions are defined on its respective scripts

"""

# Set k and n values to evaluate
k_values = [5,6,7]
n_values = [3,5, 8,10]


print('Evaluating pre-trained model embeddings...')
# Applying and savnig evaluations for pre-trained model embeddings
pre_trained_embeddings_faiss_eval(k_values,n_values)

print('Evaluating two-tower model embeddings...')
# Applying and savnig evaluations for two-tower architecture
two_towers_embeddings_eval(k_values,n_values)

print('Plotting evaluation results')

# Reading results from saved data
results_df = read_table('models_evaluation')
pt_results = results_df[(results_df['algorithm'] == 'pre-trained embeddings')]
ft_results = results_df[(results_df['algorithm'] == 'two-tower architecture')]

# Creting folders to store plots
pt_output_dir = os.path.join(PROJECT_ROOT, "models/evaluation_results/pre_trained_model")
ft_output_dir = os.path.join(PROJECT_ROOT, "models/evaluation_results/two_tower_architecture")
os.makedirs(pt_output_dir, exist_ok=True)
os.makedirs(ft_output_dir, exist_ok=True)

# Plotting and saving pre-trained embeddings results
for k in pt_results['k'].unique():
    plt.figure(figsize=(10, 6))
    subset = pt_results[pt_results['k'] == k]

    for metric in ['precision', 'recall', 'f1']:
        metric_data = subset[subset['metric'] == metric]
        plt.plot(metric_data['n'], metric_data['value'], label=metric)

    plt.xlabel('n values')
    plt.ylabel('Metrics')
    plt.title(f'Evaluation Metrics for k={k}')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(pt_output_dir, f'evaluation_pre_trained_k{k}.png')
    plt.savefig(plot_path)
    plt.close()

# Plotting and saving two-tower results
for k in ft_results['k'].unique():
    plt.figure(figsize=(10, 6))
    subset = ft_results[ft_results['k'] == k]

    for metric in ['precision', 'recall', 'f1']:
        metric_data = subset[subset['metric'] == metric]
        plt.plot(metric_data['n'], metric_data['value'], label=metric)

    plt.xlabel('n values')
    plt.ylabel('Metrics')
    plt.title(f'Evaluation Metrics for k={k}')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(ft_output_dir, f'evaluation_pre_trained_k{k}.png')
    plt.savefig(plot_path)
    plt.close()

print("Plots saved in models/evaluation_results")



