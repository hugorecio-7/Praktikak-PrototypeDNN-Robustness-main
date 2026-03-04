import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This script generates plots showing the worst-case accuracy across different attacks for each model.
def plot_worst_case_accuracy_per_model(model_names, attack_names, max_eps, step, csv_dir="results/Accuracy", output_dir="results/MinAccuracyPlot"):
    """
    For each model, this function creates a plot showing the minimum accuracy (i.e., worst-case performance)
    across all attacks for each epsilon value.

    Parameters:
        model_names (list): List of model names (strings).
        attack_names (list): List of attack names (strings).
        max_eps (float or int): Maximum epsilon value (e.g., 0.8).
        step (float or int): Step size for epsilon values.
        csv_dir (str): Directory where CSV files with results are stored.
        output_dir (str): Directory where output plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create the list of epsilon values (x-axis)
    dim = max_eps + 1 if isinstance(max_eps, int) else int(round(max_eps / step)) + 1
    x_axis = np.linspace(0, max_eps, dim) if isinstance(max_eps, float) else np.arange(0, max_eps + 1, 1)

    # Loop over each model
    for model in model_names:
        all_accuracies = []

        # For each attack, read the corresponding CSV and extract the accuracy values for this model
        for attack in attack_names:
            csv_file = os.path.join(csv_dir, f"{model}_{attack}_maxeps({max_eps})_step({step})_results.csv")
            
            if not os.path.exists(csv_file):
                print(f"Warning: {csv_file} not found.")
                continue
            
            df = pd.read_csv(csv_file, index_col=0)
            acc = df[model].values
            all_accuracies.append(acc)

        if not all_accuracies:
            print(f"No data found for model {model}. Skipping.")
            continue

        # Compute the worst-case (minimum) accuracy across all attacks for each epsilon
        min_accuracies = np.min(np.vstack(all_accuracies), axis=0)

        # Plot the worst-case accuracy curve
        plt.figure(figsize=(8, 6))
        plt.plot(x_axis, min_accuracies, label=f"{model} (Worst-case)")
        plt.scatter(x_axis, min_accuracies, s=10)
        plt.xlabel("Epsilon")
        plt.ylabel("Minimum Accuracy Across Attacks")
        plt.title(f"Worst-case Accuracy vs Epsilon\nModel: {model}")
        plt.grid(True)
        plt.legend()

        # Save plot to file
        output_path = os.path.join(output_dir, f"{model}_WorstCaseAccuracy_maxeps({max_eps})_step({step}).jpg")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved worst-case plot for model '{model}' to {output_path}")


if __name__ == "__main__":

    model_names = ["S30", "B30", "RS30", "RB30", "FTB30n", "FTB30p", "SENN_0_01", "ProtoVAE"]

    attack_names = [
    "PGDLInf_attack",
    "FSGM_attack",
    "LinfDeepFool_attack",
    "LinfAdditiveUniformNoise_attack", 
    "LinfBasicIterative_attack",
    "LinfFMNA_attack",
    "LinfMomentumIterativeFastGradient_attack",
    "LinfAdamProjectedGradientDescent_attack",
    "AutoAttack_adv",
    ]

    max_eps = 0.8
    step = 0.025

    plot_worst_case_accuracy_per_model(model_names, attack_names, max_eps, step)

