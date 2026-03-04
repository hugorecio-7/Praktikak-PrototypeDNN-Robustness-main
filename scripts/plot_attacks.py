import os
import pandas as pd
import matplotlib.pyplot as plt

# Generates plots for each attack showing accuracy vs epsilon for all models.
def plot_all_models_for_each_attack(model_names, attack_names, max_eps, step, results_dir='results/Accuracy', save_dir='results/Plot_attacks'):
    """
    Generates a plot for each attack showing accuracy vs epsilon for all models.
    
    Args:
        model_names (list): List of model names (must match CSV naming).
        attack_names (list): List of attack names (must match CSV naming).
        max_eps (float or int): Maximum epsilon used in attacks.
        step (float): Epsilon step used.
        results_dir (str): Directory containing CSV results.
        save_dir (str): Directory to save plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if isinstance(max_eps, float):
        dim = int(round(max_eps / step)) + 1
        x_axis = [round(i * step, 4) for i in range(dim)]
    else:
        dim = max_eps + 1
        x_axis = list(range(dim))

    for attack_name in attack_names:
        plt.figure(figsize=(10, 6))
        found_any = False

        for model_name in model_names:
            csv_path = os.path.join(results_dir, f"{model_name}_{attack_name}_maxeps({max_eps})_step({step})_results.csv")
            if not os.path.exists(csv_path):
                print(f"CSV not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path, index_col=0)
            acc_values = df[model_name].values
            plt.plot(x_axis, acc_values, label=model_name)
            plt.scatter(x_axis, acc_values)
            found_any = True

        if not found_any:
            print(f"No data found for attack '{attack_name}'. Skipping plot.")
            plt.close()
            continue

        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs. Epsilon\nAttack: {attack_name}")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(save_dir, f"AllModels_{attack_name}_maxeps({max_eps})_step({step})_plot.jpg")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved plot to: {plot_path}")


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

    plot_all_models_for_each_attack(
        model_names=model_names,
        attack_names=attack_names,
        max_eps=max_eps,
        step=step
    )
