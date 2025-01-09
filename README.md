# Text Clustering as Classification with LLMs

This repository contains the implementation code for the paper "[Text Clustering as Classification with LLMs](https://arxiv.org/abs/2410.00927)".

## Initial Setup

To run the code, you need to set the `--api_key` parameter to your own OpenAI Key. Please note that OpenAI may have updated their API since the last time this code was run.

## Overview

This codebase implements a text clustering approach using LLMs (Large Language Models) by converting the clustering task into a classification problem. Here's the workflow:

1. **select_part_labels.py**
- First script to run
- Takes the original dataset and randomly selects 20% of the existing labels
- Outputs `chosen_labels.json` which will be used as "seed labels" for the LLM

2. **label_generation.py**
- Depends on: `chosen_labels.json` from step 1
- Uses OpenAI's GPT-3.5 to:
  - Generate new labels for texts that don't match the given seed labels
  - Merge similar labels to reduce redundancy
- Outputs three files:
  - `{dataset}_true_labels.json`: Original cluster labels
  - `{dataset}_llm_generated_labels_before_merge.json`: Raw LLM-generated labels
  - `{dataset}_llm_generated_labels_after_merge.json`: Final merged labels

3. **given_label_classification.py**
- Depends on: The merged labels from `label_generation.py`
- Uses GPT-3.5 to classify each text into one of the generated labels.
- Outputs `{dataset}_find_labels.json` containing the classification results

4. **evaluate.py**
- Depends on: Original dataset labels and the classification results
- Evaluates the clustering performance using multiple metrics:
  - ACC (Clustering Accuracy)
  - ARI (Adjusted Rand Index)
  - NMI (Normalized Mutual Information)

The process converts a clustering problem into a classification task by:
1. Starting with a small set of known labels (20%)
2. Using LLM to generate additional relevant labels
3. Using LLM to classify texts into these labels
4. Evaluating how well this classification matches the original clustering

This approach leverages LLM's understanding of text semantics to perform clustering without traditional distance-based clustering algorithms.

## Experiments

### Step 1: Download the Dataset

First, download the dataset from the following link: [Dataset](https://drive.google.com/file/d/1TBq3vkfm3OZLi90GVH-PVNKi3fk1Vba7/view?usp=sharing), as provided by the paper [CLUSTERLLM: Large Language Models as a Guide for Text Clustering (EMNLP2023)](https://aclanthology.org/2023.emnlp-main.858/).

### Step 2: Select Part Labels

Run the following command to select the labels that will be shown to the LLM:
```bash
python select_part_labels.py
```
The selected labels will be saved in `chosen_labels.json` within the `generated_labels` folder.

### Step 3: Run the Code
Execute the script to perform the clustering process:
```bash
bash run.sh
```
This script will run the following files:
- `label_generation.py`: Generates potential labels.
- `given_label_classification.py`: Classifies the data based on the generated labels.
- `evaluate.py`: Evaluates the final clustering results.

The code will generate the following files:
- `{dataset}_llm_generated_labels_before_merge.json`
- `{dataset}_llm_generated_labels_after_merge.json`
- `{dataset}_find_labels.json`

