# 🏦 Bank Personal Loan Prediction

> Predict whether a bank customer will accept a personal loan offer using Machine Learning.

---

## 📌 Problem Statement

Banks contact thousands of customers with personal loan offers — but only **9.6%** actually accept. This wastes time, money, and resources contacting the wrong people.

**Solution:** Build a Machine Learning model that predicts which customers are likely to accept a loan offer — so the bank can target only the right customers.

---

## 🎯 What We Predict

| Output | Meaning |
|--------|---------|
| ✅ Loan Approved | Customer is likely to accept the loan |
| ❌ Loan Not Approved | Customer is likely to reject the loan |

---

## 📁 Project Structure

```
MINIPROJECT/
│
├── data/
│   ├── Bank_Personal_Loan_Modelling.xlsx  
│   ├── bank_loan.db                                              
│
├── Notebooks/
│   ├── Datacleaning.ipynb     
│   ├── EDA.ipynb                
│   ├── Training_data.ipynb           
│   └── Advanced.ipynb    
│
├── models/
│   ├── loan_model.pkl             
│   ├── scaler.pkl                 
│   └── feature_names.pkl         
│
├── app/
│   ├── api.py                     
│   └── streamlit.py           
│
├── requirements.txt               
└── README.md                     
```

---

## 📊 Dataset Information

| Detail | Info |
|--------|------|
| File | Bank_Personal_Loan_Modelling_1_.xlsx |
| Rows | 5,000 customers |
| Columns | 14 (12 after cleaning) |
| Target | Personal Loan (0 = No, 1 = Yes) |

### Columns Used for Prediction:

| Column | Description |
|--------|-------------|
| Age | Customer age |
| Experience | Years of work experience |
| Income | Annual income in $K |
| Family | Family size (1–4) |
| Education | 1=Undergrad, 2=Graduate, 3=Advanced |
| Mortgage | Mortgage amount in $K |
| CD Account | Has certificate of deposit account? (0/1) |

### Columns Removed:

| Column | Reason |
|--------|--------|
| ID | Just a row number |
| ZIP Code | Not useful for ML |
| CCAvg | Data was corrupted — all zeros after fixing |
| Securities Account | Only 0.7% importance |
| Online | Only 1.3% importance |
| CreditCard | Only 1.1% importance |

---



## 🚀 How to Run

### Step 1 — Install Packages
```bash
pip install -r requirements.txt
```

### Step 2 — Open Jupyter and Run Notebooks in Order
```bash
jupyter notebook
```
Run these one by one:
1. `notebooks/1_data_cleaning.ipynb`
2. `notebooks/2_eda.ipynb`
3. `notebooks/3_training.ipynb`
4. `notebooks/4_advanced_pandas.ipynb`

> ⚠️ Must run notebooks first — they create the database and model files.

### Step 3 — Start FastAPI (Terminal 1)
```bash
uvicorn app.api:app --reload --port 8000
```
API runs at: **http://localhost:8000**

### Step 4 — Start Streamlit (Terminal 2)
```bash
streamlit run app/streamlit_app.py
```
App runs at: **http://localhost:8501**

---

## 🖥️ App Pages

| Page | What it shows |
|------|--------------|
| 🏠 Home | Project overview, model info, top factors |
| 📊 Dashboard | Key stats, charts, insights |
| 🔍 Prediction | Enter customer details → get loan decision |
| 📋 Data View | Browse and filter the clean dataset |

---

## 🤖 Model Details

| Detail | Info |
|--------|------|
| Algorithm | Random Forest Classifier |
| Training Size | 4,000 rows (80%) |
| Test Size | 1,000 rows (20%) |
| Accuracy | **98.4%** |
| Class Imbalance Fix | class_weight='balanced' |



## 🔑 Key Findings

| Feature | Importance | Insight |
|---------|-----------|---------|
| Income | 58.8% | #1 factor — high earners accept more |
| Education | 11.4% | Advanced degree = higher acceptance |
| Family Size | 7.2% | Larger families need loans more |
| CD Account | 6.7% | CD holders have 46% acceptance rate |
| Mortgage | 4.3% | Shows financial product usage |
| Age | 4.3% | Middle-aged customers accept more |
| Experience | 4.2% | More experience = stable income |

### ✅ Loan Likely Approved When:
- Income above **$100K**
- Education level is **Advanced**
- Has a **CD Account**
- Family size is **3 or 4**

### ❌ Loan Likely Rejected When:
- Income below **$40K**
- Education is **Undergrad**
- No CD Account
- Family size is **1**

---

## ⚙️ API Usage

**Endpoint:** `POST http://localhost:8000/predict`

**Request Body:**
```json
{
  "Age": 35,
  "Experience": 10,
  "Income": 150,
  "Family": 3,
  "Education": 3,
  "Mortgage": 100,
  "CD_Account": 1
}
```

**Response:**
```json
{
  "prediction": 1,
  "result": "✅ Loan Approved",
  "confidence": "87.0%"
}
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Programming language |
| Pandas | Data cleaning and analysis |
| Scikit-learn | Machine Learning model |
| SQLite | Database for clean data |
| FastAPI | Backend prediction API |
| Streamlit | Frontend web app |
| Jupyter | Notebooks for step-by-step workflow |

---