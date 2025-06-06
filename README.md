# Praktikak

This project extends the [PrototypeDNN-Robustness](https://github.com/jon1ran/PrototypeDNN-Robustness) repository. It contains additional adversarial attacks using [Foolbox](https://github.com/bethgelab/foolbox) and [Autoattack](https://github.com/fra31/auto-attack), focused on ones constrained by the L∞ (infinity) norm to evaluate the robustness of neural network models. Also, it incorporate three new models to be tested, [ProtoVAE](https://arxiv.org/abs/2210.08151), [SENN](https://arxiv.org/pdf/1806.07538) and [Prob-PSENN](https://arxiv.org/abs/2403.13740).

## Implemented Attacks

The following L∞-norm adversarial attacks have been added:

- `LinfAdditiveUniformNoise_Attack`
- `LinfDeepFool_Attack`
- `LinfBasicIterative_Attack`
- `LinfFMN_Attack`
- `LinfMomentumIterativeFastGradient_Attack`
- `LinfAdamProjectedGradientDescent_Attack`
- `Autoattack_adv`

## Setup

The requirements to use the project are listed in the file ```requirements.txt```. It can be installed using the following command:

```
pip install -r requirements.txt
```

## Model Evaluation

In order to evaluate a models robustness in a specific attack, this must be run:

```
Command example: python run_test.py --models ProtoVAE --attacks Autoattack_adv --max_eps 0.8 --step 0.025
(1) -models: Name(s) of the model(s) you want to test. Can be a single model or a list.
(2) -attacks: Name(s) of the adversarial attack(s) to use. Must match the implemented attack class names.
(3) -max_eps: Maximum value of the perturbation ε (epsilon) used by L∞-bounded adversarial attacks. Controls the attack strength.
(4) -step: Step size for the epsilon sweep. Determines the increment between different ε values tested (e.g., 0.0, 0.025, 0.05, ..., up to max_eps).
```

After completing the test, the results will be added in the results directory. There another two directories will be found, Plot (a plot of the accuracies obtains in each epsilon) and Accuracy (the same as plot but as a csv).

Attacks hyperparameters can be adjusted changing the code in the same evaluation script, ```run_test.py```.