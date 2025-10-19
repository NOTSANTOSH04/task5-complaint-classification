# Task 5: Financial Complaint Classification using DistilBERT

**Author:** Santosh  
**Date:** October 19, 2025  
**GitHub:** NOTSANTOSH04

--

## 📋 Overview

This project implements a **text classification system** for consumer financial complaints using a fine-tuned **DistilBERT** transformer model. The system automatically categorizes complaints into 4 main product categories, achieving **95.85% accuracy** on test data.

### Problem Statement

Financial institutions receive thousands of consumer complaints daily. Manually categorizing these complaints is time-consuming and error-prone. This project automates the classification process using state-of-the-art Natural Language Processing (NLP).

## 📊 Dataset

### Source
**Consumer Financial Protection Bureau (CFPB) Consumer Complaints Database**  
Official Link: https://www.consumerfinance.gov/data-research/consumer-complaints/

### Original Dataset (CFPB)

| Metric | Value |
|--------|-------|
| **Total Complaints** | **11+ million** |
| **Date Range** | 2011 - Present |
| **Issue** | Severe class imbalance |
| **Credit Reporting** | ~70%+ of complaints |
| **Other Categories** | <10% each |

### Processed Training Dataset

The original CFPB dataset has **severe class imbalance**. For this project:

✅ **Stratified sampling** performed to create balanced dataset  
✅ **200,000 complaints** selected from 11M+ original dataset  
✅ **50,000 per class** ensuring equal representation  
✅ **Train-Val-Test split** maintaining balance across all sets  

| Metric | Value |
|--------|-------|
| Original CFPB Dataset | 11+ million complaints |
| Sampled for Training | 200,000 complaints |
| Training Set | 144,500 (72.25%) |
| Validation Set | 25,500 (12.75%) |
| Test Set | 30,000 (15%) |
| Classes | 4 (balanced) |

### Classification Categories

| Class | Category | Training Samples | % of Training |
|-------|----------|------------------|---------------|
| 0 | Credit reporting, repair, or other | 50,000 | 25% |
| 1 | Debt collection | 50,000 | 25% |
| 2 | Consumer Loan | 50,000 | 25% |
| 3 | Mortgage | 50,000 | 25% |

### Why Balanced Sampling?

**Problem with Original Data:**
- Credit reporting: ~8M+ complaints (70%+)
- Debt collection: ~500K complaints (4%)
- Consumer loans: ~300K complaints (3%)
- Mortgage: ~600K complaints (5%)

**Solution:**
- Stratified random sampling ensures equal representation
- Prevents model bias toward majority class
- Better generalization across all complaint types
- Fair evaluation metrics

- ### Sample Data (Included in Repository)

📁 **File:** `data/first_1000_rows.csv`

**Purpose:** Reference sample to demonstrate data structure  
**Note:** This is NOT the training data - just shows what the raw CFPB data looks like  
**Source:** First 1,000 rows from original CFPB dataset (unbalanced)  
**Created by:** `data/extract_1000_rows.py`


### Dataset Features

| Column | Description | Type |
|--------|-------------|------|
| Date received | Complaint submission date | Date |
| **Product** | Financial product category (TARGET) | Categorical |
| Sub-product | Specific product type | Categorical |
| Issue | Type of complaint | Text |
| **Consumer complaint narrative** | Text description (MAIN FEATURE) | Text |
| Company | Financial institution name | Categorical |
| State | Consumer's state | Categorical |
| ZIP code | Consumer's ZIP code | Numeric |
| Company response to consumer | Resolution status | Categorical |

---

## 🏗️ Model Architecture

### Base Model
**DistilBERT** (distilbert-base-uncased)
- 6-layer Transformer (vs 12 in BERT)
- 768 hidden dimensions
- 12 attention heads
- 66 million parameters
- 40% smaller than BERT
- 60% faster than BERT
- Retains 97% of BERT's performance

### Why DistilBERT?

| Metric | BERT-base | DistilBERT | Benefit |
|--------|-----------|------------|---------|
| Parameters | 110M | 66M | 40% smaller |
| Inference Speed | 1.0x | 1.6x | 60% faster |
| Performance | 100% | 97% | Minimal loss |
| Memory | 440 MB | 260 MB | Fits 6GB GPU |

### Training Configuration

Model: distilbert-base-uncased
Task: Sequence Classification (4 classes)
Max Sequence Length: 128 tokens
Batch Size: 16
Gradient Accumulation: 2 steps
Effective Batch Size: 32
Learning Rate: 2e-5
Optimizer: AdamW
Warmup Steps: 500
Epochs: 3
Mixed Precision: FP16 (enabled)
Hardware: NVIDIA RTX 3050 (4GB VRAM)
Training Time: ~45 minutes

### Model Modifications

Base DistilBERT + Classification Head
DistilBertModel (66M params)
├── Embeddings Layer
├── 6 Transformer Blocks
│ ├── Multi-Head Self-Attention
│ ├── Feed-Forward Network
│ └── Layer Normalization
└── Classification Head
├── Pre-classifier (768 -> 768)
├── Dropout (0.2)
└── Classifier (768 -> 4)


---

## 📈 Results

### Overall Performance

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 97.32% | 95.73% | **95.85%** |
| **Loss** | 0.0849 | 0.1429 | 0.1421 |

### Per-Class Performance (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Credit Reporting** | 0.9555 | 0.9559 | 0.9557 | 7,500 |
| **Debt Collection** | 0.9439 | 0.9425 | 0.9432 | 7,500 |
| **Consumer Loan** | 0.9516 | 0.9532 | 0.9524 | 7,500 |
| **Mortgage** | 0.9831 | 0.9824 | 0.9827 | 7,500 |
| **Weighted Avg** | **0.9585** | **0.9585** | **0.9585** | **30,000** |

### Training Metrics Over Epochs

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1 | 0.2064 | 93.10% | 0.1344 | 95.52% |
| 2 | 0.1153 | 96.23% | 0.1351 | 95.69% |
| 3 | 0.0849 | 97.32% | 0.1429 | 95.73% |

### Confusion Matrix Summary

- **True Positives:** 28,755 (95.85%)
- **Misclassifications:** 1,245 (4.15%)
- **Most confused:** Debt Collection ↔ Consumer Loan (2.5%)

### Key Achievements

✅ **Balanced Performance** across all classes  
✅ **High Precision** (95%+) prevents false positives  
✅ **High Recall** (95%+) catches true complaints  
✅ **Consistent Results** across train/val/test splits  
✅ **Production-Ready** with confidence scores  

---

## 🚀 Installation

### Prerequisites

- **Python:** 3.8 or higher
- **Git with Git LFS:** For cloning large model files
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 2GB free space
- **GPU:** Optional (NVIDIA with CUDA for training)

### Step 1: Install Git LFS

Windows (using chocolatey)
choco install git-lfs

macOS
brew install git-lfs

Linux
sudo apt-get install git-lfs

Initialize Git LFS
git lfs install


### Step 2: Clone Repository

Clone repository (includes 255MB model file via LFS)
git clone https://github.com/NOTSANTOSH04/task5-complaint-classification.git
cd task5-complaint-classification

Verify LFS files downloaded
git lfs ls-files

Should show: models/distilbert-model/model.safetensors

### Step 3: Create Virtual Environment (Recommended)

Windows
python -m venv venv
venv\Scripts\activate

Linux/Mac
python3 -m venv venv
source venv/bin/activate


### Step 4: Install Dependencies

Install all required packages
python -m pip install -r requirements.txt

Or install individually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers pandas numpy scikit-learn matplotlib seaborn tqdm jupyter


### Step 5: Verify Installation

Test imports
python -c "import torch; import transformers; print('✓ Installation successful!')"

Check PyTorch version
python -c "import torch; print(f'PyTorch: {torch.version}')"

Check Transformers version
python -c "import transformers; print(f'Transformers: {transformers.version}')"


### Core Dependencies

torch==2.1.0
transformers==4.35.0
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
tqdm==4.65.0
jupyter==1.0.0


---

## 💻 Usage

### Method 1: Jupyter Notebook (Recommended for Learning)

Start Jupyter
jupyter notebook

Open: notebooks/classification.ipynb
Run all cells to see complete workflow

**The notebook includes:**
1. ✅ Data Loading & Exploration
2. ✅ Class Imbalance Analysis
3. ✅ Stratified Sampling Implementation
4. ✅ Text Preprocessing Pipeline
5. ✅ Model Training with Progress Tracking
6. ✅ Evaluation & Metrics Visualization
7. ✅ Sample Predictions with Confidence Scores

### Method 2: Python Prediction Script (Quick Inference)

Run with example complaints
python predict.py


**Custom prediction:**

from predict import ComplaintClassifier

Initialize classifier (loads trained model)
classifier = ComplaintClassifier()

Predict single complaint
complaint = "There are incorrect items on my credit report."
result = classifier.predict(complaint)

Display results
print(f"Category: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.2%}")
print("\nAll Probabilities:")
for label, prob in result['all_probabilities'].items():
print(f" {label}: {prob:.2%}")


**Expected Output:**

Category: Credit reporting, repair, or other
Confidence: 96.73%

All Probabilities:
Credit reporting, repair, or other: 96.73%
Debt collection: 1.82%
Consumer Loan: 0.95%
Mortgage: 0.50%


### Method 3: Batch Prediction

import pandas as pd
from predict import ComplaintClassifier

Load classifier
classifier = ComplaintClassifier()

Load sample data
df = pd.read_csv('data/first_1000_rows.csv')
complaints = df['Consumer complaint narrative'].dropna()

Predict for multiple complaints
results = []
for complaint in complaints[:10]:
result = classifier.predict(complaint)
results.append({
'text': complaint[:50],
'predicted': result['predicted_label'],
'confidence': result['confidence']
})

Show results
results_df = pd.DataFrame(results)
print(results_df)


### Model Training Process

Load pre-trained DistilBERT

Add classification head (4 outputs)

Freeze embeddings (optional)

Fine-tune on financial complaints

Use mixed precision (FP16)

Gradient accumulation (2 steps)

Early stopping on validation loss

Save best model checkpoint


### Inference Optimization

- **Batch Processing:** Handle multiple complaints
- **CPU Optimization:** Runs efficiently without GPU
- **Mixed Precision:** FP16 inference
- **Caching:** Model loaded once, reused

### Hardware Requirements

**Minimum (Inference):**
- CPU: Any modern processor
- RAM: 4GB
- Storage: 2GB

**Recommended (Training):**
- CPU: Intel i5 / AMD Ryzen 5+
- RAM: 16GB
- GPU: NVIDIA GTX 1060 6GB or better
- Storage: 10GB

---

## 📸 Screenshots

### Dataset Class Distribution

<img width="1597" height="936" alt="image" src="https://github.com/user-attachments/assets/07b1562c-cbc7-4bbd-a9a9-361f397d8c0f" />


*Balanced distribution across all 4 categories (25% each)*

---

### Training Progress

<img width="1596" height="929" alt="image" src="https://github.com/user-attachments/assets/a24742fb-c182-4273-8e3b-01a7700b1f1c" />
<img width="1588" height="874" alt="image" src="https://github.com/user-attachments/assets/5cdfcf95-be89-425b-b5fa-f16373256004" />



*Loss and accuracy curves showing model convergence over 3 epochs*

---

### Confusion Matrix

<img width="1596" height="873" alt="image" src="https://github.com/user-attachments/assets/7694e109-432b-41be-8833-0618148cffb8" />


*Model predictions vs actual labels on 30,000 test samples*

---

### Classification Report

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a5376865-1948-4ab9-a8a1-91301ed3db1f" />


*Precision, Recall, and F1-scores achieving 95%+ across all categories*

---

### Sample Predictions

<img width="1599" height="968" alt="image" src="https://github.com/user-attachments/assets/2e2805c9-8b67-4bad-b6bb-d0cff882b50c" />


*Real-world complaint classifications with confidence scores*

---










