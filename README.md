# LLM - Zero Shot Text Classification

## Project Overview
This project implements and evaluates a Large Language Model (LLM) based text classification system, building upon recent research by Wang et al. (2023) that demonstrates the effectiveness of LLMs as zero-shot classifiers. The implementation includes a complete pipeline from data preprocessing to model evaluation, with comprehensive documentation and analysis.

## Research Background
This project is based on a key paper:
1. "Large Language Models Are Zero-Shot Text Classifiers" (Wang et al., 2023)

These papers demonstrate that LLMs can effectively perform text classification tasks without task-specific training, and can be enhanced through prompt engineering and few-shot learning approaches.

## Repository Structure
```
LLM-Assess/
├── data/                  # Raw and processed datasets
├── src/                   # Source code
│   ├── preprocess.py     # Data preprocessing and prompt creation
│   ├── train.py          # Zero-shot and few-shot implementation
│   └── evaluate.py       # Model evaluation and metrics
├── notebooks/            # Jupyter notebooks for analysis
├── models/              # Saved model checkpoints
├── evaluation_results/  # Generated evaluation metrics and plots
├── requirements.txt     # Project dependencies
└── report.pdf          # Detailed project report
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LLM-Assess.git
cd LLM-Assess
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing
```bash
python src/preprocess.py
```
This script prepares the data and creates prompt templates for zero-shot and few-shot classification.

### Model Training/Inference
```bash
python src/train.py
```
This script implements both zero-shot and few-shot classification approaches.

### Model Evaluation
```bash
python src/evaluate.py
```
This script generates comprehensive evaluation metrics and visualizations.

## Model Architecture
The project implements an LLM-based classification system with the following features:
- Zero-shot classification capabilities
- Prompt engineering for optimal performance
- Comprehensive evaluation metrics including:
  - Zero-shot
  - Classification Report

## Results
The model evaluation generates several metrics and visualizations:
- Zero-shot classification performance
- Classification report with precision, recall, and F1-score
- Confusion matrix visualization

## Project Report
A detailed report (`report.pdf`) is included in the repository, covering:
- Introduction and research background
- Related work and literature review
- Methodology and implementation details
- Experimental setup and results
- Analysis and conclusions

## Dependencies
Key dependencies include:
- PyTorch
- Transformers
- scikit-learn
- pandas
- matplotlib
- seaborn

For a complete list of dependencies, see `requirements.txt`.


## Acknowledgments
- Wang et al. for their groundbreaking research on LLMs as text classifiers
- Hugging Face Transformers library
- PyTorch team
- Original paper authors and contributors 
