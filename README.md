# Cooperative or Competitive? Understanding the Interaction between Attention Heads From A Game Theory Perspective

<!---[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)]()-->
<!---[![Arxiv](https://img.shields.io/badge/arXiv-6666.66666-B21A1B)]()-->
<!---[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-blue)]()-->


This repository provides the official PyTorch implementation of the following paper: 
> [**Cooperative or Competitive? Understanding the Interaction between Attention Heads From A Game Theory Perspective**]() <br>
> Xiaoye Qu<sup>1</sup>, 
> Zengqi Yu<sup>1</sup>,
> Dongrui Liu<sup>2</sup>,
> Wei Wei<sup>1</sup>,
> Daizong Liu<sup>2</sup>,
> Jianfeng Dong<sup>2</sup>,
> Yu Cheng<sup>2</sup><br>
> <sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>Shanghai AI Laboratory<br>
## Overview
Despite the remarkable success of attention-based large language models (LLMs), the precise interaction mechanisms between attention heads remain poorly understood.
In contrast to prevalent methods that focus on individual head contributions, we rigorously analyze the intricate interplay between attention heads by introducing a novel framework leveraging the Harsanyi dividend from cooperative game theory.
Our analysis reveals that significant positive Harsanyi dividends are sparsely distributed across head combinations, 
indicating that most heads do not contribute cooperatively. Moreover, certain head combinations exhibit negative dividends, indicating implicit competitive relationships. 
To further optimize the interactions among attention heads, 
we propose a training-free Game-theoretic Attention Calibration (GAC) method. 
Specifically, GAC selectively retains heads demonstrating significant cooperative gains and applies fine-grained distributional adjustments to the remaining heads.
Comprehensive experiments across 17 benchmarks demonstrate the effectiveness of our proposed GAC and its superior generalization capabilities across diverse model families, scales, and modalities.
Crucially, the discovered interaction phenomena offer a path toward a deeper understanding of the behaviors of LLMs.

## Installation

Follow the steps below to set up the environment and install the necessary dependencies:

1. Create a new Conda environment:

```
conda create --name GAC python=3.9
conda activate GAC
```

2. Install the required dependencies: Ensure that pip is installed, then execute the following command:

```
pip install -r requirements.txt
```

## Run GAC

### Step 1. Calculate Harsanyi Dividends Between Players for Each Layer

The first step in the GAC framework involves identifying groups of attention heads that exhibit significant positive Harsanyi dividends.

#### 1.1 Classification Task 
To perform this search on classification datasets, run the following command:

```
bash scripts/compute_game_theory_cf.sh
```

#### 1.2 Multiple Choice Task

For multiple-choice datasets, use the following command:

```
bash scripts/compute_game_theory_mc.sh
```

#### 1.3 Question Answer Task
To conduct this search on question-answering datasets, execute the following command:

```
bash scripts/compute_game_theory_qa.sh
```

#### 1.4 POPE Task
For MLLM's POPE datasets, run the following command:

```
bash scripts/compute_game_theory_pope.sh
```

### Step 2. Identify the Salient Group

After calculating the Harsanyi dividends, use the following command to select the salient group:

```
bash scripts/get_salient_group.sh [eval_task_type]
```

### Step 3. Evaluation

To mitigate the competition between attention heads that results in negative Harsanyi dividends, we apply fine-grained distributional adjustments to the heads outside the salient group. This process smooths out excessive attention weight allocations.

Run the following command to complete this process:

```
bash evaluation.sh [augmentations] [dataset_names] [model_type] [model_path] [eval_task_type]
```

## Directory Structure

Below is an overview of the key directories and files in this repository:

* model_aug:  Contains the core methods for our GAC framework.
* transformers: Includes the core code of the transformers library.
* tools: Contains code for calculating intermediate results.
* utils: Defines methods for data loading and preprocessing.
* game_theory.py: The main code for calculating game theory-related metrics.
* main.py: The central code for model evaluation.


## Datasets and Model

The datasets and models we used are as follows.

### Datasets
* sst2
* sst5
* MR
* SUBJ
* AGNews
* TREC
* CB
* BoolQ
* hellaswag
* ARCE
* PIQA
* ARCC
* OB
* CQA
* SQuADv1
* SQuADv2
* [MSCOCO 2014 dataset](https://cocodataset.org/#home)
* [GQA dataset](https://cs.stanford.edu/people/dorarad/gqa/download.html)

### Models

* [Llama-3.1-8B-Instruct
](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
* [Qwen2.5-7B-Instruct
](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
* [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
* [Qwen2.5-32B-Instruct
](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
* [llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
