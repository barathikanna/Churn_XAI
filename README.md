# 1. synthetic dataset generation and causal estimation validation using DoWhy
- This project demonstrates a complete pipeline for generating a synthetic customer churn dataset with known causal structure, estimating the treatment effect using DoWhy, and training interpretable models for churn prediction.

## 🔨 1.1. Synthetic Dataset Generation

    - Created a synthetic dataset of 10,000 customers with the following features:
      - `age`, `gender`, `visits_last_month`, `avg_purchase_value`
      - Confounder: `loyalty_score`
      - Treatment: `discount_offer`
      - Outcome: `churned`
      - Counterfactual outcome: `counterfactual_churned`
      - Natural language reviews (`complaint_text`) generated from causal variables

    - The data is causally valid: `loyalty_score` influences both treatment and outcome.
    - The outcome (`churned`) was simulated using a do-calculus-inspired process, not conditional probability.

    > Code: See `dataset.ipynb`

---

## 1.2. Causal Estimation with DoWhy
    - Used `dowhy` to:
    - Identify the **Average Treatment Effect (ATE)** of `discount_offer` on `churned`
    - Adjust for confounding using **backdoor criterion**
    - Refute the estimate with a **placebo treatment test**

        ### Key Output:
        Estimated ATE: -0.1849
        Interpretation: Discounts reduce churn by ~18.5%
        Placebo Effect: -0.00178 (non-significant)
    > causal effect is robust and not due to random correlations.

    > Code: See `dataset.ipynb`

#2. ML Model for churn prediction: 

## 2.1Linear Regression Model for DiCE (Structured data only)
    - Trained an interpretable model using only structured features (DiCE-compatible)
    - Included `class_weight='balanced'` to handle imbalance
    - Evaluated on accuracy, precision, recall, F1, and ROC AUC

    ### Without Class Balancing:
        - Precision/Recall: 0.00 (model ignored churners)

    ### With Class Balancing:
        - Accuracy : 55.85%
        - Precision : 39.53%
        - Recall : 71.29%
        - F1 Score : 50.86%
        - ROC AUC Score : 61.95%

    > Accuracy dropped but recall improved dramatically

    > Code: See `LR_DiCE.ipynb`

