# Praktikak

This project extends the [PrototypeDNN-Robustness](https://github.com/jon1ran/PrototypeDNN-Robustness) repository. It contains additional adversarial attacks using [Foolbox](https://github.com/bethgelab/foolbox) and [Autoattack](https://github.com/fra31/auto-attack), focused on ones constrained by the L∞ (infinity) norm to evaluate the robustness of neural network models. Also, it incorporate three new models to be tested, [ProtoVAE](https://arxiv.org/abs/2210.08151), [SENN](https://arxiv.org/pdf/1806.07538) and [Prob-PSENN](https://arxiv.org/abs/2403.13740).

## Implemented Attacks

The following L∞-norm adversarial attacks have been added:

- `LinfAdditiveUniformNoise_attack`
- `LinfDeepFool_attack`
- `LinfBasicIterative_attack`
- `LinfFMNA_attack`
- `LinfMomentumIterativeFastGradient_attack`
- `LinfAdamProjectedGradientDescent_attack`
- `AutoAttack_adv`

## Available models to test

The following models can be tested:

- `B30`
- `S30`
- `RS30`
- `RB30`
- `FTB30n`
- `FTB30p`
- `ProtoVAE`
- `SENN_0_01`

I could not implement Prob-PSENN for the adversarial testing.

## Setup

The requirements to use the project are listed in the file ```requirements.txt```. It can be installed using the following command:

```
pip install -r requirements.txt
```

## Model Evaluation

In order to evaluate a model robustness in a specific attack, this must be run:

```
Command example: python run_test.py --models ProtoVAE --attacks Autoattack_adv --max_eps 0.8 --step 0.025
(1) -models: Name(s) of the model(s) you want to test. Can be a single model or a list.
(2) -attacks: Name(s) of the adversarial attack(s) to use. Must match the implemented attack class names.
(3) -max_eps: Maximum value of the perturbation ε (epsilon) used by L∞-bounded adversarial attacks. Controls the attack strength.
(4) -step: Step size for the epsilon sweep. Determines the increment between different ε values tested (e.g., 0.0, 0.025, 0.05, ..., up to max_eps).
```
!Please do not use a list of models with different data bound. ProtoVAE use (-1, 1), Prob-PSENN use (-1, 1) and wider images and rest of the models use (0, 1).

After completing the test, the results will be added in the results directory. There another two directories will be found, Plot (a plot of the accuracies obtains in each epsilon) and Accuracy (the same as plot but as a csv).

Attacks hyperparameters can be adjusted changing the code in the same evaluation script, ```run_test.py```.

## Update and added Code:

### Adversarial_Attacks_Testing.ipynb
- Added tests for models and attacks.
- Includes visualizations of adversarial examples.
- Mainly exploratory, not relevant.

### adversarial_attacks.py
- Adapted FGSM and PGD to support models with input bounds in (-1, 1).
- Added new adversarial attacks using Foolbox and AutoAttack. 
- Also, for the AutoAttack a class was created to be able to use models with bounds (-1, 1)


### data_loader.py
- Updated `get_test_loader` to:
  - Transform input data to (-1, 1) range for ProtoVAE and Prob-PSENN.
  - Resize images to 32×32 for Prob-PSENN.

### model_testing.py
- Updated `adversarial_attacks_eps_plot`: 
  - Now supports Foolbox and AutoAttack via a new variable named `foolbox_use`.
- Added `adversarial_attacks_eps_plot_test`, a copy of `adversarial_attacks_eps_plot` with some changes: 
  - Supports multiple models and attacks.
  - Saves accuracy results and plots for each attack and model to the `results/` directory.
- Added `plot_adversarial_examples`: 
  - Visualizes randomly selected original images and their corresponding adversarial examples (the number is defined by the `num_examples` variable) for a given model and attack, along with their predicted labels before and after the attack.

### plot_attacks.py
- Generates accuracy vs epsilon plots for each attack with all the models at once.
- Use CSV results from `results/accuracy/`.
- Saves figures to `results/Plot_attacks/`.

### plot_models.py
- Generates minimum accuracy (worst-case for all the attacks) vs epsilon plots for each model.
- Use CSV results from `results/accuracy/`.
- Saves plots to `results/MinAccuracyPlot/`.

### run_test.py
- Main script for launching experiments I used.
- Uses `adversarial_attacks_eps_plot_test` from `model_testing.py`.
- Command-line arguments:
  - `--models`: Name(s) of the models to test (e.g., ProtoVAE, Prob-PSENN).
  - `--attacks`: Name(s) of the attacks to use (e.g., Autoattack_adv).
  - `--max_eps`: Maximum epsilon value to test.
  - `--step`: Step size between epsilon values.
- Loads models and their weights, and initializes the selected attacks with their parameters.

- Shell script containing multiple `sbatch` commands to submit adversarial attack experiments to the cluster.
- Defines model–attack combinations for batch testing.
- Calls `sbatch_experiments.sh` with the appropriate arguments for each experiment.
- Organized in blocks (e.g., `PGDLInf_attack`, `FSGM_attack`) for clarity and easy batch submission.

### experiments.sh
- Shell script containing multiple `sbatch` commands, that call `sbatch_experiments.sh` to submit adversarial attack experiments to the cluster.

### sbatch_experiments.sh
- SLURM job submission script to run experiments on a GPU cluster.
- Executes run_test.py
- Command-line arguments:
    - $1: Name of the model to test (e.g., ProtoVAE, S30).
    - $2: Name of the attack to use (e.g., PGDLInf_attack).
    - --max_eps: 0.8 (fixed)
    - --step: 0.025 (fixed)

### foolbox directory
- Contains all the Foolbox attack implementations. **No changes** were made to the original code.

### ProtoVAE directory
- Includes the trained ProtoVAE model and its corresponding model script with helper functions. 
- No modifications were made to the original repository, except for the removal of unnecessary files and the inclusion of the trained model.

### SENN directory
- Contains the trained SENN models (including the one used in the experiments), the model script, and helper functions. 
- No changes were made to the original source, only reduced to exclude unnecessary files.

### ProbP_SENN directory
- Includes the trained ProbP-SENN model along with the model script and helper functions.  
-No modifications to the original source code, only reduced to keep relevant files and the trained model.

### results
- Stores all the output results from the adversarial tests:
- **Accuracy/**: Accuracy scores for each model-attack pair across different epsilon values, saved in CSV.
- **MinAccuracyPlot/**: Plots of minimum accuracy (worst-case for all the attacks) vs epsilon for each model.
- **Plot/**: Plots of accuracy vs epsilon for a specific attack and a model.
- **Plot_attacks/**: Plots of accuracy vs epsilon for each attack with all the models at once.
