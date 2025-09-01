E-commerce Customer Churn Prediction with Explainable AI

This repository contains the full pipeline and assets for the dissertation project:

**"A Multimodal Explainable AI Framework for Customer Churn Prediction in E-commerce"**

The project combines **machine learning, causal inference, and explainable AI (XAI)** to predict customer churn and provide **transparent insights** for managers. It also includes a **Streamlit dashboard** for interactive exploration.



Features

* **Synthetic Dataset Generation** using a causal DAG with loyalty, discounts, churn, and review text.
* **Models**:
  * Logistic Regression (structured only)
  * XGBoost (structured + unstructured reviews)

* **Imbalance Handling**: Stratified sampling + SMOTE oversampling.

* **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC.

* **Interpretability**:
  * **SHAP**: Global feature attribution
  * **LIME**: Local case-by-case explanations
  * **DiCE**: Counterfactual recommendations

* **Deployment**: Interactive **Streamlit dashboard**.

repository structure

├── data/                  # Dataset and preprocessing outputs
│   └── causal_discount_churn_DAG_clean.csv
├── notebooks/             # Jupyter notebooks for data generation and experiments
├── results/               # contains all the result output and files
├── dashboard/             # Streamlit app code (app.py and supporting files)
├
├── requirements.txt       # Dependencies for reproducibility
└── README.md              # Project documentation 

## System Requirements

* **Python**: 3.10
* **Core Libraries**: NumPy, Pandas, Scikit-learn, XGBoost, Imbalanced-learn
* **Interpretability**: SHAP, LIME, DiCE
* **Causal Inference**: DoWhy
* **NLP**: scikit-learn TF-IDF
* **Deployment**: Streamlit
* **Development Environment**: Jupyter Notebook



## Installation

### Install Dependencies

```bash
pip install -r requirements.txt
```



## Reproducing the Pipeline

### Step 1: Generate Dataset

The dataset is created using a **causal DAG** (loyalty → discount → churn).
Run the dataset generation notebook:

dataset_generation.ipynb


This will produce:
`data/causal_discount_churn_DAG_clean.csv`

---

### Step 2: Train Models

Two models are trained:

* **Logistic Regression** → structured features only
* **XGBoost** → structured + TF-IDF review features

Run:

model_training.ipynb


---

### Step 3: Evaluate Models

Model performance is assessed with multiple metrics:

* **Logistic Regression (structured only)**
* **XGBoost (structured only)**
* **XGBoost (structured + reviews)**

Outputs include classification reports, ROC curves, and comparison tables.

---

### Step 4: Interpretability


interpretability.ipynb


This generates:

* **SHAP summary plots** (global drivers of churn)
* **LIME explanations** (local predictions)
* **DiCE counterfactuals** (actionable suggestions)

---

### Step 5: Launch Dashboard

The Streamlit dashboard integrates everything into one interface.

Run:

```bash
cd dashboard
streamlit run app.py
```

Dashboard modules:

* **Customer Input Panel** → enter attributes + reviews
* **Prediction Panel** → churn probability + binary output
* **Global Insights** → SHAP plots
* **Local Explanations** → LIME plots
* **Counterfactual Suggestions** → DiCE recommendations

---


## Usage Example

1. Generate dataset with synthetic reviews.
2. Train XGBoost with TF-IDF.
3. Evaluate with ROC-AUC and F1-score.
4. Interpret churn cases with LIME.
5. Use dashboard to explore customer churn in real-time.

---







