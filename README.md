
## Introduction

Deep Neural Networks (DNNs) have achieved remarkable success in various applications, but their lack of interpretability and vulnerability to adversarial attacks pose significant challenges. This project aims to address these issues by investigating the adversarial robustness of prototype-based self-explaining neural networks. These networks enhance interpretability by providing explanations based on learned prototypes, but their reliability in adversarial scenarios needs thorough evaluation.

The model implementation is based on this implementation: https://github.com/mostafij-rahman/PyTorch-PrototypeDL

## Principal Contributions

1. **Adversarial Attack Framework**: Development of a comprehensive framework for generating adversarial attacks on prototype-based neural networks. This includes various attack paradigms such as class-changing and explanation-altering attacks, both targeted and untargeted.

2. **Robust Training Methodologies**: Proposal of robust training methodologies and tailored modifications to improve the resilience of prototype-based neural networks against adversarial attacks while maintaining interpretability.

3. **Experimental Validation**: Demonstration of the low robustness of prototype-based self-explainable neural networks and the effectiveness of proposed defensive strategies through extensive experiments.

## Repository Structure

1. **Scripts**: Contains scripts to train and evaluate models.
2. **Pre-trained Models**: Already trained models are available in the `saved_model` directory.
3. **Documentation**: Detailed research is explained in the document `Investigating Adversarial Vulnerabilities and Defensive Strategies in Prototype-Based Self-Explaining Deep Neural Networks.pdf`.

## Usage

This repository is intended for researchers and practitioners in the field of machine learning and artificial intelligence who are interested in exploring the robustness and interpretability of prototype-based neural networks. Users can train new models and evaluate their robustness against adversarial attacks.

