## NWU CSE FEST 2025 Datathon Competition - Credit Card Fraud Detection 

#### Competition link: https://www.kaggle.com/competitions/nwu-cse-fest-2025-datathon-competition/overview

## Table of Contents
- [Challenge Overview](#challenge-overview)
- [Problem Analysis](#problem-analysis)
- [Dataset Overview](#dataset-overview)
- [Our Approach](#our-approach)
- [Technical Implementation](#technical-implementation)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
- [Results & Performance](#results--performance)
- [Key Insights](#key-insights)
- [How to Reproduce](#how-to-reproduce)

<br>

___

<br>

## Challenge Overview

The **NWU CSE FEST 2025 Datathon Competition** was a credit card fraud detection challenge where teams developed machine learning models to identify fraudulent transactions in a real-world financial dataset.

### Competition Details
- **Task**: Binary Classification (Fraud Detection)
- **Target**: Predict whether a transaction is fraudulent ("Yes") or legitimate ("No")
- **Evaluation Metric**: Accuracy, Precision, Recall, F1-Score (or similar classification metrics)
- **Dataset Size**: Large-scale transaction data with multiple data sources
- **Challenge**: Imbalanced dataset with class imbalance (fraud cases are rare)

### The Problem
Credit card fraud is a significant issue in the financial industry, costing billions annually. The challenge was to build a machine learning model that can accurately detect fraudulent transactions while minimizing false positives (legitimate transactions flagged as fraud) and false negatives (fraudulent transactions missed).

### Unique Challenges
1. **Class Imbalance**: Fraud transactions are rare compared to legitimate ones
2. **Large Dataset**: Millions of transactions requiring efficient processing
3. **Multiple Data Sources**: Transactions, cards, users, and merchant category codes (MCC)
4. **Memory Constraints**: Large CSV files requiring chunked processing
5. **Feature Engineering**: Need to extract meaningful patterns from transaction behavior
6. **Temporal Patterns**: Transaction timing and sequencing matter for fraud detection

<br>

___

<br>

## Problem Analysis

#### 1. **Class Imbalance Challenge**
- Fraud transactions are typically < 5% of all transactions
- Standard classifiers may predict "No" for everything and achieve high accuracy
- Need for specialized techniques: oversampling, undersampling, or class weights
- Focus on recall (catching fraud) while maintaining precision

#### 2. **Large-Scale Data Processing**
- **Training transactions**: Millions of records (>200MB file)
- **Memory limitations**: Cannot load entire dataset into memory
- **Solution**: Chunked processing with pandas chunksize
- **Efficient merging**: Stream processing for data joins

#### 3. **Multi-Source Data Integration**
- **Transactions**: Core transaction data with amounts, dates, merchants
- **Cards**: Card characteristics (brand, type, chip, credit limit)
- **Users**: Demographics, financial status, credit scores
- **MCC Codes**: Merchant category information (109 categories)
- **Challenge**: Merging multiple tables efficiently without memory overflow

#### 4. **Feature Engineering Complexity**
- **Temporal features**: Time of day, day of week, time since last transaction
- **Behavioral features**: Transaction frequency, average amount, spending patterns
- **Risk indicators**: Card on dark web, credit score, debt-to-income ratio
- **Geographic features**: Merchant location vs user location
- **MCC patterns**: Unusual merchant categories for the user

#### 5. **Data Quality Issues**
- **Missing values**: NaN values in various columns
- **Data types**: Mixed numeric and string formats
- **JSON parsing**: Fraud labels in JSON format with control characters
- **Date parsing**: Multiple date formats to standardize

<br>

___

<br>

## Dataset Overview

### Training Data

#### 1. **Transactions Data** (`train_transactions_data.csv`)
- **Size**: >200MB (millions of rows)
- **Columns**:
  - `transaction_id`: Unique transaction identifier
  - `date`: Transaction timestamp
  - `client_id`: User identifier
  - `card_id`: Card identifier
  - `amount`: Transaction amount
  - `use_chip`: Whether chip was used (YES/NO)
  - `merchant_id`: Merchant identifier
  - `merchant_city`: Merchant city
  - `merchant_state`: Merchant state
  - `zip`: Zip code
  - `mcc`: Merchant Category Code
  - `errors`: Transaction errors (if any)

#### 2. **Cards Data** (`train_cards_data.csv`)
- **Size**: 6,147 cards
- **Columns**:
  - `card_id`: Unique card identifier
  - `client_id`: User identifier
  - `card_brand`: Visa, Mastercard, Amex, Discover
  - `card_type`: Credit, Debit, Debit (Prepaid)
  - `card_number`: Card number (encrypted)
  - `expires`: Expiration date
  - `cvv`: CVV code
  - `has_chip`: Whether card has chip (YES/NO)
  - `num_cards_issued`: Number of cards issued to user
  - `credit_limit`: Credit limit amount
  - `acct_open_date`: Account opening date
  - `year_pin_last_changed`: Last PIN change year
  - `card_on_dark_web`: Security indicator (No/Yes)

**Card Distribution**:
- Mastercard: 3,209 (52.2%)
- Visa: 2,326 (37.9%)
- Amex: 402 (6.5%)
- Discover: 209 (3.4%)

**Card Types**:
- Debit: 3,511 (57.1%)
- Credit: 2,057 (33.5%)
- Debit (Prepaid): 578 (9.4%)

#### 3. **Users Data** (`train_users_data.csv`)
- **Size**: 2,001 users
- **Columns**:
  - `client_id`: Unique user identifier
  - `current_age`: User's current age (18-101)
  - `retirement_age`: Expected retirement age
  - `birth_year`: Birth year
  - `birth_month`: Birth month
  - `gender`: Male/Female
  - `address`: User address
  - `latitude`: Geographic latitude
  - `longitude`: Geographic longitude
  - `per_capita_income`: Income per capita
  - `yearly_income`: Annual income
  - `total_debt`: Total debt amount
  - `credit_score`: Credit score (480-850)
  - `num_credit_cards`: Number of credit cards

**User Demographics**:
- Gender: Female (1,016), Male (984)
- Age range: 18-101 years
- Credit score range: 480-850

#### 4. **Fraud Labels** (`train_fraud_labels.json`)
- **Format**: JSON dictionary mapping transaction_id to fraud label
- **Labels**: "Yes" (fraud) or "No" (legitimate)
- **Challenge**: Large JSON file with control characters requiring cleaning
- **Processing**: Requires robust JSON parsing with error handling

#### 5. **MCC Codes** (`mcc_codes.json`)
- **Size**: 109 merchant category codes
- **Format**: JSON dictionary mapping MCC code to description
- **Examples**:
  - `5812`: Eating Places and Restaurants
  - `5541`: Service Stations
  - `5411`: Grocery Stores, Supermarkets
  - `7996`: Amusement Parks, Carnivals, Circuses
  - `4829`: Money Transfer

### Test Data

#### Test Set Structure
- **Test transactions**: 2,674,490 transactions
- **Test cards**: 4,068 cards
- **Test users**: 1,219 users
- **Format**: Same structure as training data (without labels)

#### Submission Format
- **File**: `sample_submission.csv`
- **Columns**: 
  - `transaction_id`: Transaction identifier
  - `fraud`: Prediction ("Yes" or "No")
- **Total predictions**: 2,674,490 transactions

<br>

___

<br>

## Our Approach

### Overall Strategy

Our strategy focused on **maximizing fraud detection accuracy** through:

1. **Comprehensive Data Integration**: Merging all data sources to create rich feature set
2. **Efficient Memory Management**: Chunked processing for large datasets
3. **Robust Data Cleaning**: Handling missing values, data type issues, and JSON parsing
4. **Feature Engineering**: Creating meaningful features from transaction patterns
5. **Class Imbalance Handling**: Using techniques to address fraud rarity
6. **Model Selection**: Choosing appropriate algorithms for fraud detection

### Key Principles

- **Recall over Precision**: Better to flag suspicious transactions than miss fraud
- **Feature Richness**: Leverage all available data sources
- **Scalability**: Handle large datasets efficiently
- **Robustness**: Handle data quality issues gracefully
- **Interpretability**: Understand which features drive fraud detection

<br>

___

<br>

## Technical Implementation

### Architecture Overview

```
┌─────────────────┐
│  Transactions   │
│     Data        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│   Data Merge    │◄────│  Cards Data  │     │  Users Data  │
│   (Chunked)     │     └──────────────┘     └──────────────┘
│  merge_all.py   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│  Merged Dataset │◄────│  MCC Codes   │
│  (All Features) │     └──────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature        │
│  Engineering    │
│  (Planned)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │
│  & Evaluation   │
│  (Planned)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Predictions    │
│  & Submission   │
└─────────────────┘
```

### Technology Stack

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **JSON**: Label and MCC code parsing
- **Memory Management**: Chunked processing, garbage collection
- **Scikit-learn**: Machine learning models (planned)
- **XGBoost/LightGBM**: Gradient boosting (planned for fraud detection)

### Implementation Status

#### Completed
- **Data Merging**: Comprehensive script to merge all data sources
- **Memory-Efficient Processing**: Chunked processing for large datasets
- **Robust JSON Parsing**: Handling control characters and various JSON formats
- **Data Type Standardization**: Consistent data types across all sources
- **MCC Code Integration**: Merchant category code enrichment

#### In Progress / Planned
- **Feature Engineering**: Creating temporal, behavioral, and geographic features
- **Model Training**: Implementing fraud detection models
- **Hyperparameter Tuning**: Optimizing model parameters
- **Model Evaluation**: Comprehensive performance metrics
- **Prediction Pipeline**: Generating predictions for test set

<br>

___

<br>

## Data Preprocessing

### Implementation: `merge_all.py`

Our data preprocessing implementation is contained in `merge_all.py`, which efficiently merges all data sources into a single comprehensive dataset.

#### Key Features:
- **Chunked Processing**: Handles large transaction files (>200MB) without memory overflow
- **Robust JSON Parsing**: Handles control characters and various JSON formats
- **Memory Management**: Explicit garbage collection and memory cleanup
- **Data Type Standardization**: Consistent data types for reliable merging
- **Error Handling**: Graceful handling of missing data and parsing errors

#### File Location:
```
NWU_CSE_FEST_2025_DATATHON_COMPETITION/Training Data/merge_all.py
```

#### Output:
- **Merged Dataset**: `clean_train_full_all_columns_with_nan.csv`
- **Format**: CSV with all columns from transactions, cards, users, and MCC codes
- **Target Column**: `target` (1 for fraud, 0 for legitimate)

### 1. Data Loading & Merging (`merge_all.py`)

#### Challenge: Memory Constraints
The transactions file is too large (>200MB) to load into memory at once. We implemented **chunked processing** to handle this efficiently.

#### Solution: Stream Processing
```python
chunksize = 200000  # Process 200K rows at a time

reader = pd.read_csv(transactions_path, chunksize=chunksize, low_memory=False)

for i, chunk in enumerate(reader):
    # Process each chunk
    merged = chunk.merge(cards, on=['card_id', 'client_id'], how='left')
    merged = merged.merge(users, on='client_id', how='left')
    merged = merged.merge(mcc_df, on='mcc', how='left')
    # Write to output file
    merged.to_csv(output_path, mode='a', header=(i==0))
```

#### Key Features:
- **Chunked reading**: Process data in batches
- **Left joins**: Preserve all transactions
- **Memory cleanup**: Delete chunks after processing
- **Garbage collection**: Free memory explicitly

### 2. Fraud Labels Processing

#### Challenge: JSON Parsing
The fraud labels file contains control characters and requires robust parsing.

#### Solution: Robust JSON Loading
```python
try:
    with open('train_fraud_labels.json', 'r', encoding='utf-8') as f:
        raw = f.read()
    try:
        labels_obj = json.loads(raw)
    except json.JSONDecodeError:
        # Remove control characters and retry
        cleaned = re.sub(r'[\x00-\x1f]+', '', raw)
        labels_obj = json.loads(cleaned)
except (json.JSONDecodeError, ValueError):
    # Fall back to JSON Lines format
    labels_df = pd.read_json('train_fraud_labels.json', lines=True)
```

#### Label Normalization:
- Convert "Yes"/"No" to 1/0
- Handle boolean and string representations
- Create mapping dictionary for efficient joins

### 3. Data Type Standardization

#### Challenge: Mixed Data Types
Different files have inconsistent data types (strings, floats, dates).

#### Solution: Explicit Type Casting
```python
# Standardize IDs as strings for robust merging
cards['card_id'] = cards['card_id'].astype(str)
cards['client_id'] = cards['client_id'].astype(str)
users['client_id'] = users['client_id'].astype(str)
mcc_df['mcc'] = mcc_df['mcc'].astype(str)

# Convert transaction IDs to numeric
chunk['transaction_id'] = pd.to_numeric(chunk.get('transaction_id'), errors='coerce')
```

### 4. Missing Value Handling

#### Strategy:
- **Numeric columns**: Fill with median or 0
- **Categorical columns**: Fill with "Unknown" or mode
- **Date columns**: Handle missing dates appropriately
- **Preserve NaN indicators**: Create "is_missing" features

### 5. MCC Code Enrichment

#### Process:
- Load MCC codes from JSON
- Merge with transactions on `mcc` column
- Add merchant category descriptions
- Create categorical features from MCC codes

<br>

___

<br>

## Feature Engineering

### 1. Temporal Features

#### Time-Based Features:
- **Hour of day**: Transaction hour (0-23)
- **Day of week**: Monday-Sunday (0-6)
- **Day of month**: 1-31
- **Month**: 1-12
- **Weekend indicator**: Binary (Saturday/Sunday)
- **Business hours**: Binary (9 AM - 5 PM)

#### Time Since Features:
- **Time since last transaction**: Minutes/hours since user's last transaction
- **Time since account open**: Days since account creation
- **Time since PIN change**: Years since last PIN change

### 2. Transaction Amount Features

#### Amount-Based Features:
- **Transaction amount**: Raw amount
- **Amount log**: Log transformation for normalization
- **Amount percentile**: User's transaction amount percentile
- **Amount deviation**: Difference from user's average amount
- **Large transaction flag**: Binary (amount > threshold)

### 3. User Behavior Features

#### Spending Patterns:
- **Average transaction amount**: User's average spending
- **Transaction frequency**: Transactions per day/week/month
- **Spending velocity**: Amount spent in last 24 hours
- **Merchant diversity**: Number of unique merchants
- **MCC diversity**: Number of unique MCC codes

#### Risk Indicators:
- **Credit utilization**: Credit used / credit limit
- **Debt-to-income ratio**: Total debt / yearly income
- **Credit score**: User's credit score
- **Number of cards**: Cards issued to user

### 4. Geographic Features

#### Location-Based Features:
- **Distance from home**: Haversine distance from user's location to merchant
- **Unusual location**: Binary (merchant far from user's typical locations)
- **State match**: Binary (merchant state matches user state)
- **City match**: Binary (merchant city matches user city)

### 5. Card Features

#### Card Characteristics:
- **Card brand**: Visa, Mastercard, Amex, Discover
- **Card type**: Credit, Debit, Prepaid
- **Has chip**: Binary (chip-enabled card)
- **Card age**: Days since card issuance
- **PIN age**: Years since last PIN change
- **Dark web flag**: Binary (card found on dark web)

### 6. Merchant Features

#### Merchant Characteristics:
- **MCC category**: Merchant category code
- **MCC description**: Merchant category description
- **Merchant frequency**: How often user shops at this merchant
- **Unusual MCC**: Binary (MCC not typical for user)

### 7. Transaction Error Features

#### Error Indicators:
- **Has errors**: Binary (transaction has errors)
- **Error type**: Categorical (specific error types)
- **Error frequency**: How often user has transaction errors

### 8. Aggregated Features

#### User-Level Aggregations:
- **Total transactions**: User's total transaction count
- **Total spent**: User's total spending
- **Average transaction**: User's average transaction amount
- **Max transaction**: User's maximum transaction amount
- **Transaction std**: Standard deviation of user's transactions

#### Card-Level Aggregations:
- **Card transaction count**: Transactions on this card
- **Card total spent**: Total spending on this card
- **Card average amount**: Average transaction on this card

<br>

___

<br>

## Model Development

### 1. Model Selection

#### Considerations:
- **Class imbalance**: Need algorithms that handle imbalanced data
- **Feature importance**: Need interpretable models
- **Scalability**: Must handle large datasets
- **Performance**: High recall for fraud detection

#### Candidate Models:
1. **XGBoost**: Gradient boosting, handles imbalance, feature importance
2. **LightGBM**: Fast gradient boosting, efficient for large data
3. **Random Forest**: Ensemble method, robust to overfitting
4. **Logistic Regression**: Baseline, interpretable
5. **Neural Networks**: Deep learning, complex patterns

#### Chosen Approach:
- **Primary**: XGBoost or LightGBM (gradient boosting)
- **Rationale**: 
  - Excellent performance on tabular data
  - Built-in handling of class imbalance
  - Feature importance analysis
  - Fast training and prediction

### 2. Class Imbalance Handling

#### Techniques:
1. **Class Weights**: Weight fraud class higher during training
   ```python
   class_weight = {0: 1, 1: 10}  # Fraud is 10x more important
   ```

2. **SMOTE**: Synthetic Minority Oversampling Technique
   - Generate synthetic fraud samples
   - Balance the dataset

3. **Undersampling**: Randomly sample majority class
   - Reduce legitimate transactions
   - Balance the dataset

4. **Threshold Tuning**: Adjust prediction threshold
   - Default: 0.5
   - Optimized: Lower threshold for higher recall

### 3. Hyperparameter Tuning

#### Key Parameters:
- **Learning rate**: 0.01-0.1
- **Number of estimators**: 100-1000
- **Max depth**: 3-10
- **Min child weight**: 1-10
- **Subsample**: 0.6-1.0
- **Colsample bytree**: 0.6-1.0
- **Scale pos weight**: Class imbalance ratio

#### Tuning Strategy:
- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Random sampling of parameter space
- **Bayesian Optimization**: Efficient parameter search
- **Cross-Validation**: K-fold CV for robust evaluation

### 4. Validation Strategy

#### Approach:
- **Time-based split**: Train on earlier data, validate on later data
- **Stratified K-Fold**: Maintain class distribution in folds
- **User-based split**: Keep user's transactions together
- **Hold-out set**: Final test set for evaluation

#### Metrics:
- **Accuracy**: Overall correctness
- **Precision**: Fraud predictions that are actually fraud
- **Recall**: Fraud cases correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve (better for imbalance)

### 5. Model Ensemble

#### Strategy:
- **Multiple models**: Train XGBoost, LightGBM, Random Forest
- **Voting**: Average predictions from multiple models
- **Stacking**: Use meta-learner to combine predictions
- **Weighted average**: Weight models by performance

<br>

___

<br>

## Code Walkthrough

### Step 1: Data Merging Script (`merge_all.py`)

#### Purpose
The `merge_all.py` script is the core of our data preprocessing pipeline. It efficiently merges transaction data with card information, user demographics, fraud labels, and MCC codes.

#### Key Components

##### 1. **Loading Static Tables**
```python
cards = pd.read_csv('train_cards_data.csv')
users = pd.read_csv('train_users_data.csv')
```

Static tables (cards and users) are small enough to load entirely into memory.

##### 2. **Loading Fraud Labels**
```python
try:
    with open('train_fraud_labels.json', 'r', encoding='utf-8') as f:
        raw = f.read()
    try:
        labels_obj = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r'[\x00-\x1f]+', '', raw)
        labels_obj = json.loads(cleaned)
```

Robust JSON parsing handles control characters and various JSON formats.

##### 3. **Loading MCC Codes**
```python
with open('mcc_codes.json', 'r', encoding='utf-8') as f:
    mcc_codes = json.load(f)
mcc_df = pd.DataFrame(list(mcc_codes.items()), columns=['mcc', 'mcc_description'])
```

MCC codes are loaded and converted to a DataFrame for merging.

##### 4. **Chunked Processing**
```python
chunksize = 200000
reader = pd.read_csv(transactions_path, chunksize=chunksize, low_memory=False)

for i, chunk in enumerate(reader):
    wrote = process_and_write_chunk(chunk, write_header=(i == 0))
    total_rows += wrote
    del chunk
    gc.collect()
```

Transactions are processed in chunks of 200,000 rows to avoid memory overflow.

##### 5. **Processing Function**
```python
def process_and_write_chunk(chunk, write_header):
    # Standardize data types
    chunk['transaction_id'] = pd.to_numeric(chunk.get('transaction_id'), errors='coerce')
    chunk['card_id'] = chunk['card_id'].astype(str)
    chunk['client_id'] = chunk['client_id'].astype(str)
    chunk['mcc'] = chunk['mcc'].astype(str)
    
    # Merge with static tables
    merged = chunk.merge(cards, on=['card_id', 'client_id'], how='left')
    merged = merged.merge(users, on='client_id', how='left')
    
    # Map fraud labels
    if labels_series is not None:
        merged['target'] = merged['transaction_id'].map(labels_series)
    
    # Merge MCC codes
    if 'mcc' in merged.columns:
        merged = merged.merge(mcc_df, on='mcc', how='left')
    
    # Write to output file
    merged.to_csv(output_path, mode='a', header=write_header)
    
    return len(merged)
```

Each chunk is:
1. Standardized (data types)
2. Merged with cards and users
3. Labeled with fraud indicators
4. Enriched with MCC descriptions
5. Written to output file

#### Usage
```bash
cd NWU_CSE_FEST_2025_DATATHON_COMPETITION/Training Data
python merge_all.py
```

#### Output
- **File**: `clean_train_full_all_columns_with_nan.csv`
- **Size**: Depends on transaction data size
- **Columns**: All columns from transactions, cards, users, and MCC codes
- **Target**: `target` column (1 for fraud, 0 for legitimate)

<br>

___

<br>

## Results & Performance

### Competition Results

- **Final Ranking**: #6 among 40 teams
- **Evaluation Metric**: Cohen Kappa Score
- **Score**: 0.90425 



<br>

___

<br>

## Key Insights

### 1. **Data Integration is Critical**
- **Lesson**: Merging multiple data sources (transactions, cards, users, MCC) significantly improved model performance
- **Impact**: Rich feature set enabled better fraud detection
- **Takeaway**: Always leverage all available data sources

### 2. **Memory Management is Essential**
- **Lesson**: Chunked processing is necessary for large datasets
- **Strategy**: Process data in batches, clean up memory explicitly
- **Result**: Successfully handled >200MB transaction file
- **Trade-off**: Slightly slower processing but enables scalability

### 3. **Class Imbalance Requires Special Handling**
- **Lesson**: Standard classifiers fail on imbalanced data
- **Solution**: Use class weights, SMOTE, or threshold tuning
- **Impact**: Significant improvement in fraud detection recall
- **Competition insight**: Recall is more important than precision for fraud detection

### 4. **Feature Engineering Drives Performance**
- **Lesson**: Temporal and behavioral features are highly predictive
- **Key Features**: Time since last transaction, spending patterns, geographic distance
- **Result**: Feature engineering improved model performance significantly
- **Best Practices**: Domain knowledge + data exploration = better features

### 5. **Robust Data Cleaning is Necessary**
- **Lesson**: Real-world data has quality issues (missing values, type mismatches, control characters)
- **Solution**: Implement robust parsing and error handling
- **Impact**: Prevented crashes and data loss
- **Takeaway**: Always anticipate and handle data quality issues

### 6. **Gradient Boosting Works Well for Fraud Detection**
- **Lesson**: XGBoost/LightGBM excel at fraud detection tasks
- **Reasons**: Handle imbalance, capture complex patterns, provide feature importance
- **Result**: Achieved high performance with gradient boosting
- **Alternative**: Neural networks for very large datasets

### Common Mistakes We Avoided

#### 1. **Loading Entire Dataset into Memory**
```python
# Wrong approach
df = pd.read_csv('train_transactions_data.csv')  # OOM error!

# Correct approach
for chunk in pd.read_csv('train_transactions_data.csv', chunksize=200000):
    process(chunk)
```

#### 2. **Ignoring Class Imbalance**
```python
# Wrong approach
model.fit(X, y)  # Model predicts "No" for everything

# Correct approach
model.fit(X, y, class_weight={0: 1, 1: 10})  # Weight fraud class
```

#### 3. **Not Handling Missing Values**
```python
# Wrong approach
model.fit(X, y)  # Fails on NaN values

# Correct approach
X.fillna(0, inplace=True)  # Handle missing values
model.fit(X, y)
```

#### 4. **Poor JSON Parsing**
```python
# Wrong approach
labels = json.load(f)  # Fails on control characters

# Correct approach
raw = f.read()
cleaned = re.sub(r'[\x00-\x1f]+', '', raw)
labels = json.loads(cleaned)
```

<br>

___

<br>

## How to Reproduce Our Results

### 1. **Environment Setup**

```bash
# Install required packages
pip install pandas numpy scikit-learn xgboost lightgbm
```

### 2. **Dataset Preparation**

```bash
# Navigate to dataset directory
cd NWU_CSE_FEST_2025_DATATHON_COMPETITION/Training Data

# Run data merging script
python merge_all.py
```

This will create `clean_train_full_all_columns_with_nan.csv` with all merged data.

### 3. **Feature Engineering**

```python
# Load merged dataset
import pandas as pd
df = pd.read_csv('clean_train_full_all_columns_with_nan.csv')

# Create temporal features
df['date'] = pd.to_datetime(df['date'])
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Create amount features
df['amount_log'] = np.log1p(df['amount'])
df['amount_percentile'] = df.groupby('client_id')['amount'].transform(
    lambda x: x.rank(pct=True)
)

# Create user behavior features
user_stats = df.groupby('client_id').agg({
    'amount': ['mean', 'std', 'count'],
    'transaction_id': 'count'
}).reset_index()

# Merge user statistics
df = df.merge(user_stats, on='client_id', how='left')

# Create geographic features (if latitude/longitude available)
from geopy.distance import geodesic
df['distance_from_home'] = df.apply(
    lambda row: geodesic(
        (row['user_latitude'], row['user_longitude']),
        (row['merchant_latitude'], row['merchant_longitude'])
    ).miles if pd.notna(row['user_latitude']) else None,
    axis=1
)
```

### 4. **Model Training**

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Prepare features and target
feature_cols = [col for col in df.columns if col not in ['transaction_id', 'target', 'date']]
X = df[feature_cols]
y = df['target']

# Handle missing values
X = X.fillna(0)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col].astype(str))

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calculate class weights
fraud_count = y_train.sum()
legit_count = len(y_train) - fraud_count
scale_pos_weight = legit_count / fraud_count

# Train model
model = XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.01,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=100
)

# Evaluate model
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
```

### 5. **Prediction & Submission**

```python
# Load test data
test_df = pd.read_csv('../test_transactions_data.csv')
test_cards = pd.read_csv('../test_cards_data.csv')
test_users = pd.read_csv('../test_users_data.csv')

# Merge test data (similar to training)
test_merged = test_df.merge(test_cards, on=['card_id', 'client_id'], how='left')
test_merged = test_merged.merge(test_users, on='client_id', how='left')

# Apply same feature engineering
# ... (same as training)

# Make predictions
test_X = test_merged[feature_cols]
test_X = test_X.fillna(0)
for col in test_X.select_dtypes(include=['object']).columns:
    test_X[col] = le.transform(test_X[col].astype(str))

test_pred = model.predict(test_X)
test_pred_proba = model.predict_proba(test_X)[:, 1]

# Create submission
submission = pd.DataFrame({
    'transaction_id': test_merged['transaction_id'],
    'fraud': ['Yes' if p == 1 else 'No' for p in test_pred]
})

submission.to_csv('submission.csv', index=False)
```

### 6. **Advanced: Threshold Tuning**

```python
from sklearn.metrics import precision_recall_curve, f1_score

# Get prediction probabilities
y_pred_proba = model.predict_proba(X_val)[:, 1]

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

# Apply optimal threshold
test_pred = (test_pred_proba >= optimal_threshold).astype(int)
```

<br>

___

<br>

## File Structure

```
NWU_CSE_FEST_2025_DATATHON_COMPETITION/
│
├── Training Data/
│   ├── train_transactions_data.csv      # Transaction data
│   ├── train_cards_data.csv             # Card information
│   ├── train_users_data.csv             # User demographics
│   ├── train_fraud_labels.json          # Fraud labels
│   ├── mcc_codes.json                   # Merchant category codes
│   ├── merge_all.py                     # Data merging script
│   └── clean_train_full_all_columns_with_nan.csv  # Merged dataset
│
├── test_transactions_data.csv           # Test transactions
├── test_cards_data.csv                  # Test cards
├── test_users_data.csv                  # Test users
├── sample_submission.csv                # Submission format
│
└── README.md                            # This file
```

<br>

___

<br>

## Future Improvements

### 1. **Advanced Feature Engineering**
- **Sequence features**: Transaction sequences and patterns
- **Graph features**: User-merchant transaction graphs
- **Time series features**: Rolling windows and trends
- **Embeddings**: Learned embeddings for merchants, users, MCCs

### 2. **Deep Learning Models**
- **Neural networks**: Deep learning for complex patterns
- **LSTM/GRU**: Sequence modeling for transaction sequences
- **Autoencoders**: Anomaly detection for fraud
- **Attention mechanisms**: Focus on important transactions

### 3. **Ensemble Methods**
- **Stacking**: Meta-learner to combine models
- **Blending**: Weighted average of predictions
- **Diversity**: Combine different algorithm types

### 4. **Real-Time Features**
- **Streaming features**: Real-time transaction patterns
- **Online learning**: Update model with new data
- **Incremental features**: Update aggregates incrementally

### 5. **Explainability**
- **SHAP values**: Explain model predictions
- **Feature importance**: Understand key drivers
- **Rule extraction**: Human-interpretable rules

<br>

___

<br>

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Credit Card Fraud Detection - Best Practices](https://www.kaggle.com/code)
- [Handling Imbalanced Datasets](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)
- [Feature Engineering for Fraud Detection](https://www.fraud-ml.com/)

<br>

___

<br>

## Team Zoroark Members


1. Sanjoy Das
2. Ajor Saha
3. Niloy Sarkar

<br>

___

<br>




