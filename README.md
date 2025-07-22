# [] LogiBreak

This repository contains a comprehensive framework for evaluating and analyzing jailbreak attempts on language models across multiple languages (English, Chinese, and Dutch). The framework consists of three main components: reformulation, jailbreak, and evaluation.

## Overview

The project implements a systematic approach to:
1. Reformulate potentially harmful requests into formal logical forms
2. Attempt jailbreaks using the reformulated requests
3. Evaluate the success of jailbreak attempts using multiple judges

## Components

### 1. Reformulation (`reformulate_*.py`)
- Reformulates potentially harmful requests into formal logical forms
- Available for multiple languages:
  - English (`reformulate_en.py`)
  - Chinese (`reformulate_zh.py`)
  - Dutch (`reformulate_du.py`)
- Uses GPT-3.5-turbo by default for reformulation
- Supports multiple restarts for each request

### 2. Jailbreak (`jailbreak_v1_*.py`)
- Attempts to jailbreak target models using reformulated requests
- Available for multiple languages:
  - English (`jailbreak_v1_en.py`)
  - Chinese (`jailbreak_v1_zh.py`)
  - Dutch (`jailbreak_v1_du.py`)
- Uses a formal semantics approach to generate jailbreak attempts
- Supports parallel processing with multiple restarts

### 3. Evaluation (`evaluate_*.py`)
- Evaluates jailbreak attempts using multiple judges:
  - Rule-based evaluation
  - GPT-4 evaluation
  - Llama3-70b evaluation
- Available for multiple languages:
  - English (`evaluate_en.py`)
  - Chinese (`evaluate_zh.py`)
  - Dutch (`evaluate_du.py`)
- Generates comprehensive evaluation results

## Usage


### Running the Pipeline

1. **Reformulation**:
```bash
python reformulate_en.py --reformulate_model gpt-3.5-turbo --n_restarts 5
```

2. **Jailbreak**:
```bash
python jailbreak_v1_en.py --target_model gpt-3.5-turbo --input_path <path_to_reformulated_queries> --n_restarts 5
```

3. **Evaluation**:
```bash
python evaluate_en.py --evaluate_llama3 False --evaluate_gpt True --input_path <path_to_jailbreak_output> --n_restarts 5
```

### Output Files
- Reformulated queries are saved in `./output/reformulated_queries/`
- Jailbreak attempts are saved in `./output/jailbreak_output/`
- Evaluation results are saved alongside the input files with an `-evaluation_result.json` suffix

## Project Structure
```
.
├── api.py                 # API interface for language models
├── judges.py             # Evaluation judges implementation
├── reformulate_*.py      # Reformulation scripts for different languages
├── jailbreak_v1_*.py     # Jailbreak scripts for different languages
├── evaluate_*.py         # Evaluation scripts for different languages
└── output/               # Output directory for results
    ├── reformulated_queries/
    └── jailbreak_output/
```

## Notes
- The framework supports multiple languages (English, Chinese, Dutch)
- Each component can be run independently
- Results are saved in JSON format for easy analysis
- The system uses multiple judges for comprehensive evaluation
- Parallel processing is implemented for efficient jailbreak attempts

