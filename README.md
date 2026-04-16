# Subscription-Based-Customer-Retention-System – A Business‑Focused Machine Learning Project

## What is this project?

A telecommunications company wants to know which customers are likely to stop using their service (this is called **churn**). If they can predict churn early, they can offer discounts or better plans to keep the customer. This project builds a machine learning model that looks at customer data (like monthly charges, contract type, internet service, etc.) and estimates the probability that a customer will leave.

The goal is not just to build a model, but to make **business decisions** based on the model’s output. The company has to balance two costs:

- **Cost of a false alarm** – offering a discount to a customer who was never going to leave (wasted money).
- **Cost of a missed churner** – losing a customer forever (lost revenue).

We tune the model to find the best trade‑off.

## Why is this important?

Without a churn model, the company would either:
- Give discounts to everyone (expensive).
- Give discounts to no one (lose many customers).

With a model, they can target only the customers who are most likely to leave, saving money and keeping more customers.

## What data do we use?

We use the **Telco Customer Churn** dataset (a public dataset from IBM). It contains information about 7043 customers, including:

- Demographics (senior citizen, partner, dependents)
- Account information (tenure, contract type, paperless billing, payment method)
- Services used (phone, internet, online security, streaming TV, etc.)
- Monthly and total charges
- Whether the customer churned (Yes / No)

The dataset is imbalanced: about 73% of customers did not churn, and 27% churned. This imbalance is important – a model that simply says “no one churns” would be 73% accurate but completely useless.

## How did we build the model?

We followed these steps (you can see the full code in the repository):

### 1. Data cleaning and preparation
- Removed `customerID` and `gender` (not useful for prediction).
- Converted `TotalCharges` from text to numbers (some entries were empty – we filled them with the median).
- Split the data into training (80%) and testing (20%), keeping the same churn ratio in both.

### 2. Preprocessing
- **Numerical columns** (tenure, MonthlyCharges, TotalCharges) were scaled to have mean 0 and variance 1 (StandardScaler).
- **Categorical columns** (contract type, payment method, etc.) were converted to numbers using One‑Hot Encoding.

### 3. Feature selection
We trained a temporary Random Forest model on the transformed data and selected the **12 most important features**. This reduces noise and speeds up training.

### 4. Handling imbalance
The dataset has more non‑churners than churners. If we train directly, the model will learn to ignore churners. We used **SMOTE** (Synthetic Minority Oversampling) plus **Random Oversampling** to create a **balanced training set.**

### 5. Hyperparameter tuning with a custom business score
We trained eight different types of models:
- Logistic Regression (linear)
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- AdaBoost
- SVM with RBF kernel
- K‑Neighbors

For each model, we searched for the best hyperparameters. But we did **not** use standard accuracy. Instead, we used a **custom scoring function** that lets us choose how much we care about recall (catching churners) versus precision (avoiding false alarms).

You can adjust the variable `RECALL_WEIGHT` (between 0 and 100). For example:
- `RECALL_WEIGHT = 80` → the model will prioritize catching churners, even if that means more false alarms.
- `RECALL_WEIGHT = 30` → the model will prioritize not wasting discounts, even if it misses some churners.

In our final run, we set `RECALL_WEIGHT = 50` (equal weight to recall and precision). This gives a balanced model.

### 6. Results of individual models (after tuning)

Here is a summary (sorted by recall for churn):

| Model               | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---------------------|----------|-------------------|----------------|-------------|
| Logistic Regression | 0.7566   | 0.5278            | **0.7861**      | 0.6316      |
| AdaBoost            | 0.7665   | 0.5422            | 0.7727         | 0.6373      |
| SVM (RBF)           | 0.7594   | 0.5335            | 0.7460         | 0.6221      |
| XGBoost             | 0.7743   | 0.5601            | 0.6979         | 0.6214      |
| CatBoost            | 0.7899   | 0.5903            | 0.6818         | 0.6328      |
| K‑Neighbors         | 0.7509   | 0.5242            | 0.6658         | 0.5866      |
| LightGBM            | 0.7878   | 0.5969            | 0.6176         | 0.6071      |
| Random Forest       | 0.7871   | 0.6000            | 0.5936         | 0.5968      |

**Observation:**  
- **Logistic Regression** achieves the highest recall (78.6%) – it catches the most churners, but its precision is moderate (52.8%), meaning it also flags many loyal customers.  
- **Random Forest** has the highest precision (60.0%) – when it says “churn”, it is most often correct, but it misses about 40% of actual churners (recall only 59.4%).  
- There is a clear trade‑off: models that are good at finding churners (high recall) tend to produce more false alarms (lower precision), and vice versa.  

This trade‑off is exactly why we use **threshold tuning** – we can adjust the model’s sensitivity after training to match the business cost of a false alarm versus a missed churner.

### 7. Stacking – combining models to get a “manager”

We took three diverse models (Logistic Regression, AdaBoost, SVM) and combined them using a **stacking classifier**. A meta‑model (Logistic Regression) learns how to best weigh their predictions.

**What we found:** The stacked model became more conservative. It increased precision but lowered recall compared to the best individual models. This is expected – the manager waits for consensus before raising an alarm.

**Do we need stacking?**  
Not necessarily. Stacking gives a model that wastes fewer retention discounts, but if our business priority is to catch as many churners as possible (even at the cost of some wasted discounts), we can simply deploy a single bold model like Logistic Regression or AdaBoost. Stacking is an extra layer of complexity that may not be needed if you are comfortable with a single model’s trade‑off.

In this project, we keep stacking as an option, but the final business decision can use either the stack or a single model.

### 8. Threshold tuning – the “sensitivity dial”

Even after training a model, we can change its behavior without retraining. Every model outputs a probability (e.g., “70% chance this customer will churn”). By default, we predict “churn” if probability ≥ 50%. But we can move that line.

**Example: low threshold (20%)**  
The model will say “churn” even when it is only 20% sure. This catches almost all real churners (high recall) but also flags many loyal customers (low precision). This is good if missing a churner is very expensive.

**Example: high threshold (80%)**  
The model only says “churn” when it is very sure (80% or more). This rarely bothers loyal customers (high precision) but may miss some churners (low recall). This is good if false alarms are very expensive.

In our code, you can change the variable `CHOSEN_THRESHOLD` (e.g., 0.35, 0.50, 0.70) and immediately see new results without retraining.

**Threshold tuning vs. scoring parameter**  
You may wonder: why do we have both?  
- **Scoring parameter** (`scoring=` in hyperparameter search) changes the model **during training**. It makes the model’s internal logic favour recall or precision. This is a deep change, but it is slow (requires retraining).  
- **Threshold tuning** is a **fast post‑processing** adjustment. You can change it in one second.  

In real business, we often do both: first train a well‑balanced model using a metric like F1‑score, then use threshold tuning to quickly adapt to changing costs (e.g., a new marketing budget).

For this project, you can rely mainly on threshold tuning – it gives you a remote control over the model’s behaviour.

### 9. Final threshold selection (example)

We set `CHOSEN_THRESHOLD = 0.35` (a moderately low threshold) to catch more churners. Results on the test set:

| Class       | Precision | Recall | F1‑Score | Support |
|-------------|-----------|--------|----------|---------|
| No Churn    | 0.91      | 0.71   | 0.80     | 1035    |
| Churn       | 0.50      | 0.81   | 0.62     | 374     |
| **Accuracy** |           |        | **0.74 (74%)** | **1409** |

- **Out of all customers who actually churned**, the model caught **81%** (recall for Churn = 0.81).
- **Out of all customers the model flagged as “churn”**, only **50%** actually churned (precision for Churn = 0.50). The other half were false alarms (loyal customers incorrectly warned).
- This threshold gives a high recall (good for catching churners) at the cost of moderate precision (some wasted discounts).

**Business impact summary:**
- ✅ Churners caught (True Positives): 285 (20.3%)
- ❌ Churners missed (False Negatives):  86 (6.1%)
- 📢 False alarms (False Positives):    248 (17.6%)
- 😊 Loyal customers left alone (True Negatives): 788 (56.0%)

Out of all actual churners, the model caught **77%** (recall). Out of all customers the model flagged as “churn”, **53%** actually churned (precision).

The company can now decide: is 77% recall enough? If they want 85% recall, they can lower the threshold further (e.g., 0.25). If false alarms are too many, they can raise the threshold (e.g., 0.50).

## What is the ceiling of this model?

Even with the best algorithms, real‑world churn models rarely achieve recall and precision above 0.85. **The limitation is not the technology – it is the data.**

Customer behaviour has a random component. A customer may churn because they move to another city, lose their job, or have a family emergency – none of which is captured in the data. No amount of clever machine learning can predict these external events.

To push the model higher, you would need **different data**, such as:
- Customer service call transcripts (sentiment analysis)
- Real‑time app usage logs (did they suddenly stop using a key feature?)
- Social media activity (are they complaining publicly?)

Even then, some churn will always be unpredictable. That “unpredictable residue” is why 100% perfect prediction is impossible.

## How to run this project on your own laptop (Windows)

### Step 1 – Install Python and required libraries

You need Python 3.8 or higher. Then install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm catboost
```

### Step 2 – Download the dataset

The dataset is included in the repository under

```bash
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

If you clone the repository, it will be there.

### Step 3 – Open the project in VS Code or any editor

```bash
git clone https://github.com/AkshatModi870/Subscription-Based-Customer-Retention-System.git
```
```bash
cd telco-churn-prediction
```

### Step 4 – Run the Python script

The main code is in churn_model.py. Execute it:

```bash
python churn_model.py
```

You will see:

 - Data exploration output (value counts, statistics)
 - Plots (scatter plots, histograms, box plots)
 - Hyperparameter tuning results for each model
 - Stacking classifier results
 - Threshold tuning output with confusion matrices

All graphs will pop up in separate windows.

### Step 5 – Experiment with the threshold

Open the script and change the variable CHOSEN_THRESHOLD near the end. Run again and see how recall and precision change. Try values like 0.2, 0.5, 0.8.

## What do the attached images explain?

The images you see in this repository contains detailed explanations about the following concepts:

- **Why stacking may not be necessary** – stacking makes the model more conservative; sometimes a single bold model (like Logistic Regression or AdaBoost) is better for catching churners.

- **Threshold tuning vs. scoring parameter** – two ways to control the trade‑off between recall and precision. Threshold tuning is fast (you can change it in one second), while the scoring parameter changes the model internally during training (slow but deeper).

- **The ceiling of current models** – the limitation is not the algorithms; it is the data. Even the best model cannot predict churn caused by events outside the data (e.g., a customer moving to another city).

- **The smoke alarm analogy** – a simple way to understand recall and precision. A sensitive alarm catches all fires (high recall) but gives many false alarms (low precision). A strict alarm gives few false alarms (high precision) but may miss some fires (low recall).

These concepts are already summarised in this README. The images are kept for reference.

## Summary of business decisions

- **If you want to catch almost every churner (high recall)**  
  Use a low threshold (e.g., `0.2`) or train a model with `scoring='recall'`.  
  *Result*: You will waste some discounts on loyal customers (more false alarms), but you will rarely miss a churner.

- **If you want to avoid wasting discounts (high precision)**  
  Use a high threshold (e.g., `0.7`) or train with `scoring='precision'`.  
  *Result*: You will miss some churners (lower recall), but almost every customer you warn actually churns – so you waste very few discounts.

- **The best approach for most businesses**  
  Start with a balanced model (optimise for **F1‑score** during hyperparameter tuning). Then, as business costs change, **adjust only the threshold** – this is fast and does not require retraining.

This project gives you the tools to make that decision based on the company's specific cost of a false alarm versus the cost of losing a customer.
