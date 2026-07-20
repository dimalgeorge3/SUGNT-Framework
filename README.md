# Self-Uncertainty Guided Neural Training (SUGNT)

A PyTorch framework implementing Self-Uncertainty Guided Neural Training (SUGNT) with BERT, utilizing Monte Carlo Dropout and dynamic early-exits for efficient sentiment analysis.

## Overview
Large Language Models like BERT achieve state-of-the-art performance but suffer from high computational overhead during inference. This project addresses this by introducing an early-exit mechanism into a 12-layer BERT Base architecture. By estimating predictive uncertainty, the model dynamically halts inference for "easy" samples at earlier layers, saving computational resources without significantly sacrificing accuracy. 

The model is fine-tuned for binary sentiment analysis using an Amazon product review dataset.

## Core Architecture & Methodology
* **Base Model:** 12-layer BERT Base architecture.
* **Early-Exit Branches:** Classifiers attached at Layers 4, 8, and 12.
* **Uncertainty Estimation:** Monte Carlo (MC) Dropout ($p=0.1$) is applied across 3 stochastic forward passes to compute predictive variance and estimate true model uncertainty.
* **Agent-Based Controller:** A custom controller evaluates early-exit viability using a **Multi-Factor Score**, which calculates:
  1. Predictive Uncertainty
  2. Prediction Stability
  3. Model Confidence

## Dataset
* **Source:** Amazon Product Reviews (Binary Sentiment)
* **Training Set:** 10,000 samples
* **Testing Set:** 3,000 samples

## Performance & Efficiency Results
The implementation successfully demonstrates that dynamic computation can drastically reduce inference costs while maintaining high predictive performance.

* **Accuracy:** 91.47%
* **F1-Score:** 0.9037
* **Efficiency:** Achieved a **peak theoretical efficiency gain of 63.72%**.
* **Exit Distribution:** 86.40% of review samples were confidently classified and dynamically exited at Layer 4, bypassing the need to compute the remaining 8 layers.

## Requirements
To run this project, you will need the following dependencies:

```bash
pip install torch transformers datasets numpy scikit-learn
