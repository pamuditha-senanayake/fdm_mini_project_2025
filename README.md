# FDM Mini Project 2025 – Retail Dataset Analysis & Machine Learning

## Project Overview
This is a collaborative **FDM Mini Project 2025** conducted as part of the **Fundamentals of Data Mining** course at the Sri Lanka Institute of Information Technology.  
The project focuses on analyzing a real-world retail dataset, performing **data cleaning, feature engineering, exploratory data analysis (EDA)**, and building **machine learning models** (regression and classification).  
A baseline **recommendation system** is also implemented.  

The project aims to demonstrate practical application of **data mining and machine learning techniques** on a large, real-world dataset.  

---

## Dataset
- **Source:** [OpenDataBay – Retail Dataset](https://www.opendatabay.com/data/consumer/327c5b3c-9f40-45bb-a79b-d5e2c9abc68a)
- **Rows:** ~300,000
- **Columns:** ~30 (Transaction_ID, Customer_ID, Amount, Order_Status, Product_Category, etc.)
- **Format:** CSV
- **Notes:** - Dataset contains null values and duplicates  
  - Covers retail transactions from Mar 2023 to Feb 2024  
  - Stored in `data/raw/new_retail_data.csv`  

> **Important:** Keep `data/raw` read-only. Processed/cleaned data is saved in `data/processed/`.

---
## Project Setup

### Prerequisites
- Python 3.10 or higher
- Git installed on your system
- PyCharm IDE (recommended) or any Python IDE

### 1. Download & Install PyCharm

1. Visit the [PyCharm Official Website](https://www.jetbrains.com/pycharm/download/)
2. Choose **Community Edition** (free) or **Professional Edition**
3. Download the installer for your operating system (Windows/macOS/Linux)
4. Run the installer and follow the setup instructions
5. Launch PyCharm and configure your Python interpreter

### 2. Clone the Repository

Open your terminal (macOS/Linux) or Git Bash (Windows) and run the following commands:

```bash
# Navigate to your projects directory
cd ~/projects

# Clone the repository using HTTPS
git clone https://github.com/your-username/fdm-mini-project-2025.git

# OR clone using SSH (if configured)
git clone git@github.com:your-username/fdm-mini-project-2025.git

# Enter the project directory
cd fdm-mini-project-2025

2. Install dependencies:

```bash
pip install -r requirements.txt
```
### 3. Open Project in PyCharm

1. Open PyCharm
2. Go to `File → Open`
3. Select the cloned folder `fdm-mini-project-2025`
4. Configure the Python interpreter if not automatically detected

### 4. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# OR if using conda
conda env create -f environment.yml
conda activate fdm-project

```
---

## File/Folder Structure

3. Check the dataset in `data/raw/data.csv`.

4. Run Jupyter notebooks in order:

   1. `01_data_loading.ipynb`
   2. `02_data_cleaning.ipynb`
   3. `03_feature_engineering.ipynb`
   4. `04_eda.ipynb`
   5. `05_regression_model.ipynb`
   6. `06_classification_model.ipynb`
   7. `07_baseline_recommender.ipynb`

5. Processed data and trained models will be saved in `data/processed/` and `artifacts/` respectively.

---

## Python Scripts

* `scripts/preprocess.py` → Automates data cleaning and feature engineering
* `scripts/train_regression.py` → Trains regression model to predict transaction amount
* `scripts/train_classification.py` → Trains classification model to predict order status
* `scripts/recommend.py` → Generates baseline Top-N product recommendations

---

## Machine Learning Models

* **Regression:** Predicts `amount` using `HistGradientBoostingRegressor`
* **Classification:** Predicts `order_status` using `LogisticRegression`
* **Baseline Recommender:** Top-N products per customer segment or category

Saved in `artifacts/`:

* `model_reg_amount.pkl`
* `model_cls_order_status.pkl`

---

## Collaboration Guidelines

* Branch strategy:

  * `main` → Stable code & final deliverables
  * `dev/<name>` → Each member works independently
* Use pull requests to merge into `main`
* Avoid committing raw data; use `.gitignore` to exclude large files
* Optional: Use **Git LFS** for dataset if needed

---

## Deliverables

1. **SOW Document** (`reports/SOW.pdf`)
2. **Final Report** (`reports/final_report.pdf`)
3. **10-min Presentation** (`reports/presentation.mp4`)
4. **Cleaned Data & Models** (`data/processed/`, `artifacts/`)

---

## Contact

* **Team Name:** FDM Mini Project Team 2025
* **Instructor:** [Faculty of Computing – SLIIT](https://www.sliit.lk)
* **Team Members:**
    * S.M.P.B.Senanayake
    * Member 2
    * Member 3

---

## References

* Dataset: [OpenDataBay – Retail Dataset](https://www.opendatabay.com/data/consumer/327c5b3c-9f40-45bb-a79b-d5e2c9abc68a)
* Pandas Documentation: [https://pandas.pydata.org](https://pandas.pydata.org)
* Scikit-learn Documentation: [https://scikit-learn.org](https://scikit-learn.org)
```
