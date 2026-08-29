# M5 Forecasting Raw Data: Competition Context & Files

This directory must contains the original, immutable data files for the M5 Forecasting - Accuracy competition. Understanding the distinction between the "Validation" and "Evaluation" files is key to reproducing the competition phases.

Link to download: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

## File Descriptions

### 1. `calendar.csv`
Granularity: date (yyyy-mm-dd)
Contains the date-to-day mapping (e.g., `d_1` is 2011-01-29) and exogenous variables:
* **Events**: National and cultural holidays (Super Bowl, Eid al-Fitr, etc.).
* **SNAP**: Binary indicators for state-level food assistance programs.

### 2. `sell_prices.csv`
Granularity: store_id , item_id and wm_yr_wk (week)
Weekly prices for each product in each store. Prices vary over time and across different store locations.

### 3. `sales_train_validation.csv`
Granularity: store_id , item_id and d_* (day)
Contains daily sales from **d_1 to d_1913**.

### 4. `sales_train_evaluation.csv`
Granularity: store_id , item_id and d_* (day)
Contains daily sales from **d_1 to d_1941**. This is a superset of the validation file.

### 5. `sample_submission.csv`
Template for the 28-day forecast required for the competition.

---

## The Competition Logic: Validation vs. Evaluation

In the original Kaggle competition, these two files represented the two distinct stages of the challenge:

### Phase 1: The Validation Stage (Public Leaderboard)
* **Objective**: Predict sales for days **d_1914 to d_1941**.
* **Data**: Participants were only given `sales_train_validation.csv` (up to d_1913).
* **The "Blind" Window**: The ground truth for days 1914-1941 was hidden. Submissions were scored against these hidden values to rank participants on the **Public Leaderboard**.

### Phase 2: The Evaluation Stage (Final Ranking / Private Leaderboard)
* **Objective**: Predict sales for a *new* 28-day window: **d_1942 to d_1969**.
* **Data**: At this stage, `sales_train_evaluation.csv` was released. It revealed the previously hidden sales (1914-1941), allowing participants to retrain their models with the most recent data before the final forecast.
* **The Final Score**: The accuracy on days 1942-1969 determined the final ranking (Private Leaderboard).

## Research Application

For this study on **Explainable AI (XAI)**, we use the `sales_train_evaluation.csv` file. 

By treating **d_1 to d_1913** as our training set and **d_1914 to d_1941** as our test/audit set, we can simulate the competition's first phase while having the "ground truth" labels available to calculate error metrics and evaluate the **fidelity and stability** of SHAP and LIME when the model encounters prediction errors.