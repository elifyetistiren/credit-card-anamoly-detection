# Credit-Card-Anamoly-detection

## 📝 Project Background and Overview

With the rapid growth of digital transactions, credit card fraud has become a pressing issue for financial institutions and payment processors. Detecting fraudulent activity in real time is essential to minimize financial losses and protect customer trust.

This project utilizes anonymized credit card transaction data collected over two days, encompassing over **280,000 records**. The primary aim is to build a machine learning model capable of accurately identifying potentially fraudulent transactions. By analyzing patterns in transaction behavior, time, and engineered features, the goal is to develop a robust, data-driven approach that enhances fraud detection capabilities while minimizing false positives.

This project investigates credit card transaction data to uncover fraud patterns and improve detection accuracy. It analyzes transactional behavior across time, amount ranges, and anonymized features to identify suspicious activity. By exploring fraud frequency, transaction value, and temporal trends, the analysis pinpoints high-risk periods and behavioral anomalies that help guide targeted prevention strategies.

Leveraging **Python** for data wrangling, modeling, and visualization, the project applies both **supervised** (XGBoost, Logistic Regression) and **unsupervised** (Isolation Forest, One-Class SVM) algorithms. Among these, a tuned **XGBoost classifier**—optimized via cross-validation and hyperparameter search—demonstrates superior performance in identifying fraud. Furthermore, the decision threshold is fine-tuned post-training to maximize **recall** while preserving high **precision**, minimizing missed frauds without overwhelming the system with false alarms.

Evaluation includes industry-standard performance metrics, with a focus on **AUC-ROC** and **Precision-Recall (PR)** curves. While ROC-AUC measures overall classification power, PR-AUC is emphasized due to the extreme class imbalance in fraud detection, offering a more sensitive measure of the model’s ability to identify rare fraud cases.

An interactive dashboard complements the analysis by surfacing actionable fraud insights for stakeholders. It tracks core fraud KPIs and model performance:

### 📊 Fraud Distribution KPIs
- **Fraud Count by Hour** – Frequency of fraud by hour of day.  
- **Fraud Rate by Hour** – Percentage of hourly transactions that are fraudulent.  
- **Fraud Amount by Hour** – Total value of fraud over time intervals.

### 💰 Transaction-Based KPIs
- **Fraud Amount Distribution** – Spread of fraud values across transactions.  
- **Fraud Count by Amount Range** – Bucketed transaction values to highlight fraud-prone ranges.

### 📈 Model Evaluation KPIs
- **AUC-ROC & PR-AUC** – Classification strength with an emphasis on recall under class imbalance.  
- **Threshold Optimization** – Custom tuning of the XGBoost decision threshold to improve fraud recall.  
- **Precision, Recall, F1-Score** – Key metrics guiding model selection.  
- **Feature Importance** – Top predictors of fraud surfaced through model interpretation.

This end-to-end fraud detection solution not only uncovers vulnerabilities in transaction data but also empowers stakeholders with actionable insights and model-backed risk indicators through a streamlined, code-driven analytical workflow.

---

## 🧰 Technology Stack

- **Jupyter Notebooks**  
  Used for exploratory data analysis and model development.  
  Notebooks include visualizations and markdown commentary to explain insights and modeling decisions.

- **Python Scripts**  
  Modular scripts handle data extraction (`data_processing.py`), model training (`train_model.py`), and scoring (`predict_new.py`).  
  Designed for reusability and production readiness.

- **SQL Queries**  
  Includes `.sql` files or embedded queries for data analysis and feature generation (e.g., fraud trends, transaction stats by class).  
  Highlights proficiency in SQL for fraud data mining.

- **Power BI Dashboard**  
  An interactive dashboard visualizes key metrics such as fraud distribution, transaction volume over time, and feature-based insights.  
  Built from cleaned data exported from Python or SQL.

- **Environment & Tools**
  - **Languages**: Python 3, SQL  
  - **Libraries**: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `seaborn`, optional `xgboost`  
  - **Database**: MS Access  
  - **Dashboard**: MS Power BI  
  - **Setup**: `requirements.txt` and optional MS Access database initializer provided for easy environment setup

---

## ⚙️ Setup

- **Install Dependencies**  
  - Use the `requirements.txt` file to install all necessary Python packages using `pip`.

- **Download the Dataset**  
  - Download the credit card fraud detection dataset from Kaggle.  
  - Place the CSV file into a folder named `data/` in the project directory.

- **Run the Project**  
  - Launch Jupyter Notebook and open `FraudDetection.ipynb` to explore the data.  
  - Review the saved outputs if already run.

- **Review the Dashboard**  
  - Open `dashboard.png` or the provided `.pbix` file to view the Power BI fraud dashboard.

---
## 📚 Data Structure

**Kaggle's Credit Card Fraud Detection Dataset**  
This dataset contains transactions made by European cardholders in September 2013.  
It includes **284,807 transactions**, with only **492 labeled as fraudulent**, making it a highly **imbalanced dataset**.

### Key Table

- **`creditTransactions`** – Serves as the primary table containing transaction-level data.  
  It includes time-based information, anonymized features (`V1` through `V28`), transaction amount, and a binary flag indicating fraud status.

### Data Dictionary

| **Column Name** | **Data Type** | **Description** |
|-----------------|---------------|-----------------|
| `Time`          | Float         | Seconds elapsed between each transaction and the first recorded transaction |
| `V1 - V28`      | Float         | Principal components (PCA-transformed features for confidentiality) |
| `Amount`        | Float         | Transaction amount in Euros |
| `Class`         | Integer       | Indicator of fraud (`0 = Legitimate`, `1 = Fraudulent`) |

---

## 🧠 Model Methodology

This project combines **supervised classification** and **unsupervised anomaly detection** techniques to build a robust credit card fraud detection system.

### 1. Data Preprocessing
- Loaded and cleaned the dataset (`creditcard.csv`) containing 284,807 transactions.
- Handled data issues:
  - Removed 1,081 duplicate rows.
  - No missing values were present.
- Performed data transformations:
  - Log-transformed `Amount` → `Log_Amount` to normalize right-skewed distribution.
  - Converted `Time` (in seconds) into derived features:
    - `Hour`: Time of day (0–23).
    - `Day`: Day of the week (0–6).
- Converted column data types (`int8`, `float32`) to optimize memory usage.
- Addressed extreme **class imbalance** (fraud cases ~0.17%) using:
  - `class_weight="balanced"` in models like Logistic Regression.
  - `scale_pos_weight` in XGBoost.

### 2. Modeling Techniques

### Logistic Regression (Baseline)
- A simple linear classifier to establish a baseline.
- Used `StratifiedKFold` cross-validation for fair evaluation across imbalanced classes.
- Applied `class_weight="balanced"` to penalize underrepresented fraud class.
- **Performance:**
  - High overall accuracy (~98%) due to imbalance.
  - Very low precision for fraud class (~6%), though recall was high (~91%).
  - Low F1-score (~0.11) for fraud class — indicated it wasn't suitable alone.

### XGBoost Classifier (Primary Model)
- A powerful tree-based ensemble model using gradient boosting.
- Configured with:
  - `scale_pos_weight` to handle class imbalance.
  - `eval_metric='logloss'` for probability calibration.
- Used in multiple phases:
  - Default parameters for baseline comparison.
  - Hyperparameter tuning (see section below).
  - Threshold optimization to improve fraud recall vs. precision trade-off.
- **Performance after tuning:**
  - Precision (fraud): ~99%
  - Recall (fraud): ~79%
  - F1-score (fraud): **0.88**
  - ROC-AUC and PR-AUC showed strong model performance.
- Feature importance visualized: `V17`, `V14`, `V12` ranked highest.

### Isolation Forest (Unsupervised Anomaly Detection)
- Learns the pattern of normal transactions and isolates outliers.
- Set contamination rate to 0.0017 (matching fraud prevalence).
- Trained only on `X_train` and evaluated on test data.
- **Performance:**
  - Precision: ~20%
  - Recall: ~23%
  - F1-score: ~0.22
  - Poor at accurately identifying fraud; high false positives.

### One-Class SVM (Unsupervised)
- Kernel-based method (RBF) to learn boundary around normal transactions.
- Trained using only non-fraud samples.
- Parameters:
  - `nu=0.0017` (expected fraction of anomalies).
  - `gamma='auto'`.
- **Performance:**
  - Precision: ~3%
  - Recall: ~81%
  - F1-score: ~0.05
  - Extremely high false positive rate → low practical usability.

---

## 🔧 Hyperparameter Tuning

### XGBoost Hyperparameter Optimization

- **Goal**: Maximize **F1-score** of fraud class using `RandomizedSearchCV`.
- **Cross-validation**: 3-fold `StratifiedKFold`.
- **Search Space & Best Parameters**:

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "gamma": [0, 0.1, 0.2]
    }

    best_params = {
        "subsample": 0.8,
        "n_estimators": 200,
        "max_depth": 7,
        "learning_rate": 0.2,
        "gamma": 0.2,
        "colsample_bytree": 0.8
    }

### Retraining & Final Evaluation

- Retrained XGBoost using the best hyperparameters on 80% training data.
- Evaluated on 20% test data.
- **Results (Optimized Model):**
  - Precision (fraud): **0.99**
  - Recall (fraud): **0.79**
  - F1-score (fraud): **0.88**
  - Confusion Matrix:

        [[56648     3]
         [   20    75]]

### Threshold Optimization

- Tested thresholds from 0.1 to 0.9.
- Tracked F1-score for each threshold.
- Found **optimal threshold = 0.7** (maximized F1).
- **Improved Results**:
  - Precision: **0.99**
  - Recall: **0.79**
  - F1-score: **0.88**


---

## 🚀 Executive Summary

### Overview of Findings

This credit card fraud detection analysis provides key insights into transaction behaviors, highlighting fraud trends and peak times while emphasizing important transaction features. The goal was to develop a machine learning model capable of accurately identifying fraudulent transactions while minimizing false positives.

Given the highly imbalanced nature of the dataset, special techniques were employed to optimize model performance, particularly in fraud detection. The analysis reveals that fraudulent transactions, though rare, follow distinct patterns that can be leveraged for improved fraud detection and prevention.

- **Fraud Rate**: Fraudulent transactions account for **0.17%** of the total. Although rare, this low rate poses a significant challenge for accurate detection.
- **Fraudulent Transaction Amounts**: On average, fraudulent transactions tend to involve higher amounts than legitimate ones.  
  The majority of fraudulent activity occurs under **$10**, but fraud rates spike for transactions **above $500**.


### Key Insights

- **Fraud Peaks by Time**:  
  Fraud activity peaks at **2 AM (1.71%)**, with additional spikes at **11 AM, 12 PM, and 6 PM**.  
  These patterns suggest fraud is more likely during certain hours of the day.

- **Fraud by Day**:  
  Fraud occurs more frequently on **Day 0 (0.19%)** compared to **Day 1 (0.15%)**, indicating a slight decrease in fraud over time.

- **Amount-Related Fraud Trends**:  
  - While most transactions are of lower value, fraudulent ones are more evenly distributed across the amount range.  
  - Fraud rates increase significantly for amounts **over $500**.

- **Key Fraud Indicators**:  
  Features such as V4, V14, and V12 show strong associations with fraudulent transactions.  
  Notably, V17 has the highest correlation (0.301) with fraud, making it a critical feature for detection.

- **Model Evaluation**:  
  XGBoost, with a fine-tuned decision threshold of 0.70, emerged as the top-performing model.  
  It achieved a precision of 99% and recall of 79%, effectively balancing accuracy and false positive reduction.

- **ROC-AUC**:  
  The model scored a high ROC-AUC of 0.98, demonstrating strong capability in distinguishing fraudulent from legitimate transactions.

  ---

## 🕵️‍♂️ Exploratory Data Analysis (EDA) and Insights

### 1. Transaction Amount Distribution
- The original transaction amounts are highly right-skewed, with most values at the lower end and a few large outliers. This skew can negatively impact model performance.
- A **log transformation** was applied to normalize the distribution, improving model suitability by reducing the impact of outliers.  
  
### 2. Class Imbalance
- The dataset shows a **severe class imbalance**, with **99.83%** non-fraudulent transactions and only **0.17%** fraudulent transactions.  
  A basic model might achieve high accuracy by predicting only non-fraudulent cases, which would be misleading.
- This imbalance necessitates **careful model handling** to ensure sufficient focus on the minority class (fraudulent transactions).

### 3. Transaction Amounts and Fraud
- **Fraudulent transactions** are more evenly distributed across different amounts.  
  In contrast, **non-fraudulent transactions** are mostly concentrated in lower amounts.
- As the transaction amount increases, the **fraud rate also increases**, especially for amounts **over $500**, which show significantly higher fraud rates.

### 4. Feature Distribution
- Several features, including **V4, V10, V12, V14, and V17**, display clear separation between fraud and non-fraud distributions, indicating potential as strong fraud indicators.
- Features like **V1, V7, and V13** exhibit noticeable skewness in the fraudulent class, which could benefit models sensitive to distribution shape.
- **Principal Component Analysis (PCA)** reveals that certain features (e.g., V4, V17) have high correlations with fraud, reinforcing their usefulness in fraud detection.

### 5. Fraud Trends by Time
- **Fraud Rate by Hour**: Fraud spikes notably at **2 AM (1.71%)** and remains elevated between **2–5 AM**.  
  Other moderate peaks occur at **11 AM, 12 PM**, and **6 PM**, while fraud is less frequent during mid-morning and late-night hours.
- **Fraud Rate by Day**:  
  - **Day 0**: Higher fraud rate at **0.19%**  
  - **Day 1**: Slightly lower fraud rate at **0.15%**

### 6. Feature Correlation
- Features like **V4, V14, V12**, and **V17** show **strong correlations with fraud**.  
  For example, **V17** has a correlation of **0.301**, highlighting its significant predictive power.
- The **Amount** feature shows little correlation with fraud, suggesting fraud detection depends on more complex feature interactions.

### 7. Fraud Detection Insights
- **Fraudulent Transaction Amount by Hour**: Fraudulent transactions occur across all hours, with no clear peak for large-value fraud.
- **Fraudulent Transactions by Amount Range**:  
  Most fraudulent transactions involve amounts **under $10**, but fraud rates rise significantly for transactions **above $500**.

### 8. Fraud Count by Time of Day
- The highest number of fraudulent transactions occurs in the **evening (54.12%)**, followed by the **morning (15.01%)**.  
  Fraud is least frequent in the **afternoon (12.94%)** and **night (11.68%)**.

### 9. Summary Statistics
- **Mean Transaction Amount (Non-Fraudulent)**: `$88.29`  
- **Mean Transaction Amount (Fraudulent)**: `$122.21`  
- **Overall Fraud Rate**: `0.17%` of transactions are fraudulent

---
## 💡 Recommendations

Based on the uncovered insights, the following recommendations have been provided to improve fraud detection and prevention strategies:

- **Focus on High-Risk Transaction Hours**  
  Given the fraud spikes observed between **2 AM and 5 AM**, it is recommended to implement stricter fraud detection measures during these hours.  
  Introducing real-time monitoring with increased alert thresholds could help prevent fraud during these critical periods.

- **Enhance Fraud Detection for Higher Transaction Amounts**  
  Fraud rates increase significantly as transaction amounts rise, particularly for amounts **over $500**.  
  It is advised to apply additional scrutiny or implement **multi-factor authentication (MFA)** for large transactions, as they tend to exhibit higher fraud risks.

- **Feature Utilization**  
  Features like **V4, V14**, and **V12**, which show strong separation between fraudulent and non-fraudulent transactions, should be prioritized in the model.  
  These features are **critical fraud indicators** and can help refine detection models for better accuracy.

- **Transaction Timing Optimization**  
  Fraud detection algorithms should incorporate **time-of-day patterns**, with heightened focus on **evening hours** when fraudulent activity peaks.  
  Targeted alerts during these periods could significantly reduce fraud-related losses.

- **Threshold Adjustment for Optimal Sensitivity**  
  Adjusting the fraud detection threshold to optimize for **recall** while maintaining **precision** is essential.  
  Based on model performance, a **threshold of 0.70** is recommended to balance the trade-off between detecting fraud and reducing false positives, thereby improving overall accuracy.

- **Refining False Positive Management**  
  With a **precision of 99%** and **recall of 79%** at the optimized threshold, it is advised to continue closely monitoring false positives.  
  Regular adjustments to the decision threshold should be made in response to evolving fraud patterns, ensuring a sustained balance between detection accuracy and user experience.


---

## 🏁 Conclusion

This credit card fraud detection project uncovered key fraud patterns and successfully applied machine learning to detect fraud.  
Insights from the **exploratory data analysis (EDA)** highlighted fraud peaks during specific hours, higher rates for larger transactions, and critical features strongly linked to fraudulent behavior.  
These findings informed the development of an **optimized model** that balances **precision and recall**, enabling effective fraud detection with **minimal false positives**.

The recommendations—such as targeting high-risk hours and optimizing detection thresholds—provide **actionable strategies** for financial institutions to enhance fraud prevention and reduce potential losses.

Implementing this project in a financial setting could improve **real-time fraud detection**, guide **risk mitigation strategies**, and significantly reduce fraud-related **financial impact**.  
This underscores the **real-world value** of using data analysis and machine learning to safeguard financial transactions, promoting **greater security** and **operational efficiency**.

Ultimately, this project addresses real-world challenges and offers a **meaningful contribution to the financial industry**.


