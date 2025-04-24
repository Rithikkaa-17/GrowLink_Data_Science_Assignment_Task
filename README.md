Credit Card Fraud Detection

Grow Link | Data Science Assignment | Task 5 
Candidate Name:Rithikkaa S J  

This project aims to detect fraudulent credit card transactions using various machine learning models.Given the highly imbalanced nature of real-world fraud detection problems, the project focuses on balancing data effectively and comparing model performance across multiple metrics.

Project Overview

Credit card fraud results in significant financial losses annually. This project develops and evaluates classification models to predict fraud accurately, minimizing false positives while ensuring high sensitivity.

Objective:
Build an efficient binary classification model to detect fraudulent transactions using real-world anonymized credit card data.

Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Samples:284,807 transactions  
- Features:  
  - Time and Amount 
  - V1 to V28 : PCA-transformed features  
  - Class: Target variable (0 = Non-fraud, 1 = Fraud)

Note: The dataset is highly imbalanced (fraud: ~0.17%).

Preprocessing

- Removed low-correlation features (correlation < 0.10).
- Handled missing values using `dropna()` on the target column.
- Balanced the dataset using:
  - Under-sampling (RandomUnderSampler)
  - Over-sampling (SMOTE)
- Performed stratified train-test split.

Models Used

1. Random Forest
2. XGBoost
3. AdaBoost
4. Gradient Boosting
5. Voting Classifier (Ensemble of Random Forest + XGBoost)

 Evaluation Metrics

We evaluated models using:
- Accuracy
- Precision, Recall, F1-score (Macro & Weighted)
- ROC AUC Score

 Results Summary

| Model              | Accuracy | F1 (Weighted) | ROC AUC | F1 (Macro) |
|-------------------|----------|---------------|---------|------------|
| Random Forest      | 97.65%   | 98.56%        | **0.9880** | 59.63%     |
| XGBoost            | 99.61%   | 99.68%        | 0.9645  | 80.41%     |
| AdaBoost           | 89.38%   | 94.09%        | 0.9402  | 49.88%     |
| Gradient Boosting  | 89.36%   | 94.08%        | 0.9398  | 49.87%     |
| Voting Classifier  | **99.80%** | **99.82%**  | 0.9727  | **87.45%** |

- Best by ROC AUC: Random Forest  
- Best Overall Performance: Voting Classifier

 Visualizations

Key insights were drawn using:
- Confusion matrices
- ROC curves
- Probability distributions
- Class balance graphs
- Feature importance visualizations

How to Run

 On Google Colab
- Open the notebook: `Grow_Link_Data_Science_Assignment.ipynb`
- Upload the dataset or use Kaggle API
- Run all cells and explore outputs

Locally
```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
jupyter notebook

 
