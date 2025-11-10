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
- **Evaluation Metric**: Cohen Kappa Score
- **Dataset Size**: 
  - Training: 6,240,474 transactions
  - Test: 2,674,489 transactions
- **Challenge**: Extreme class imbalance (fraud rate: ~0.15%)

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

#### 1. **Extreme Class Imbalance Challenge**
- **Fraud rate**: 0.15% (9,332 fraud cases out of 6,240,474 transactions)
- Standard classifiers may predict "No" for everything and achieve high accuracy but low Kappa
- **Solution**: Used `scale_pos_weight` in XGBoost to handle imbalance
- **Threshold Optimization**: 2-phase threshold tuning to maximize Cohen Kappa score
- **Evaluation**: Cohen Kappa score rewards balanced predictions (not just accuracy)

#### 2. **Large-Scale Data Processing**
- **Training transactions**: 6,240,474 records
- **Test transactions**: 2,674,489 records
- **Memory**: ~7.6 GB training data, ~2.6 GB test data
- **Solution**: Used Parquet format for efficient I/O, indexed by transaction_id
- **GPU Acceleration**: Leveraged XGBoost GPU for fast training (~14 minutes for 5-fold CV)

#### 3. **Multi-Source Data Integration**
- **Transactions**: Core transaction data with amounts, dates, merchants
- **Cards**: Card characteristics (brand, type, chip, credit limit)
- **Users**: Demographics, financial status, credit scores
- **MCC Codes**: Merchant category information (109 categories)
- **Challenge**: Merging multiple tables efficiently without memory overflow

#### 4. **Feature Engineering Complexity**
- **Temporal features**: Hour, day of week, weekend flag, night flag, month
- **Account features**: Account age (days), PIN age (years)
- **Amount features**: Log amount, high amount flag, amount per day
- **Geographic features**: Vectorized Haversine distance from home (10x faster)
- **Target encoding**: Fraud rates for card_id, client_id, merchant_id, mcc, merchant_state, use_chip
- **Frequency encoding**: Transaction counts for card_id, client_id, merchant_id, merchant_state
- **Risk interactions**: Amount × merchant risk, distance × merchant risk, MCC risk × amount
- **Data drift**: Identified use_chip and has_chip as 100% missing in test (dropped)

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

Our strategy focused on **maximizing Cohen Kappa Score** through:

1. **Comprehensive Feature Engineering**: Created 47 features from transaction patterns, user behavior, and merchant characteristics
2. **Target Encoding**: Leveraged fraud rates for cards, clients, merchants, and MCC codes
3. **Frequency Encoding**: Captured transaction frequency patterns
4. **Geographic Features**: Vectorized Haversine distance calculation for merchant-user distance
5. **Temporal Features**: Extracted time-based patterns (hour, day, weekend, night)
6. **Risk Interactions**: Created merchant risk × amount, distance × merchant risk interactions
7. **XGBoost with GPU**: Leveraged GPU acceleration for fast training
8. **Threshold Optimization**: 2-phase threshold tuning (coarse → fine) for optimal Kappa
9. **Drift Handling**: Identified and dropped columns with 100% missing values in test set

### Key Principles

- **Kappa Optimization**: Focused on Cohen Kappa score as the evaluation metric
- **Target Encoding**: Used fraud rates for high-cardinality categoricals (merchant_id, mcc, card_id)
- **Feature Interactions**: Created risk-aware interactions (amount × merchant risk, distance × merchant risk)
- **Data Drift Awareness**: Identified and handled test-time drift (use_chip, has_chip 100% missing in test)
- **Efficient Processing**: Vectorized operations for speed (10x faster distance calculation)
- **Regularization**: Strong L1/L2 regularization to prevent overfitting on imbalanced data

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
- **Data Preprocessing**: Indexed datasets, dropped PII, cleaned money columns, parsed dates
- **Feature Engineering**: Created 47 features (temporal, geographic, target encoding, frequency encoding, interactions)
- **Model Training**: XGBoost with GPU, 5-fold Stratified K-Fold, threshold optimization
- **Prediction Pipeline**: Generated predictions with optimal threshold (0.717)
- **Submission Generation**: Created submission file with 2,674,489 predictions

#### Key Achievements
- **Final Score**: Cohen Kappa = 0.90425
- **CV Performance**: Mean CV Kappa = 0.89627, OOF Kappa = 0.89522
- **Ranking**: #6 out of 40 teams
- **Feature Count**: 47 engineered features
- **Training Time**: ~14 minutes on GPU

<br>

___

<br>

## Data Preprocessing

### Implementation Overview

Our data preprocessing pipeline was implemented in a Kaggle notebook and consisted of multiple stages:

#### Stage 1: Data Loading & Indexing
- Loaded pre-merged Parquet files (training: 6.24M rows, test: 2.67M rows)
- Set `transaction_id` as index for efficient lookups
- Verified uniqueness: All transaction IDs are unique (no duplicates)

#### Stage 2: PII Removal
- Dropped sensitive columns: `card_number`, `cvv`, `address`, `merchant_city`
- Preserved anonymized identifiers: `card_id`, `client_id`, `merchant_id`

#### Stage 3: Data Cleaning
- **Money columns**: Removed `$` and commas, converted to numeric (amount, credit_limit, income, debt)
- **Dates**: Parsed date columns (date, acct_open_date, expires, year_pin_last_changed)
- **PIN year**: Extracted year from year_pin_last_changed, dropped original column
- **Fraud labels**: Converted "Yes"/"No" to 1/0 (int8)

#### Stage 4: Categorical Handling
- Converted categorical columns to pandas Category dtype
- Added 'missing' category for handling NaN values
- Standardized categories across train and test

#### Stage 5: Data Type Optimization
- Downcasted float64 → float32 (memory savings)
- Downcasted int64 → int32/int16/int8 where possible
- Optimized memory usage: Train reduced to ~1.3 GB, Test to ~0.4 GB

#### Stage 6: Missing Value Imputation
- **Numeric columns**: Filled with median values
- **Categorical columns**: Filled with 'missing' sentinel value
- **Result**: Zero missing values in final dataset

#### Output:
- **Cleaned Datasets**: `train_clean.parquet`, `test_clean.parquet`
- **Format**: Parquet with transaction_id as index
- **Features**: 33 columns after cleaning (32 in test, no fraud column)

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

Our feature engineering pipeline created **47 features** from the cleaned dataset. Here's the complete breakdown:

### 1. Temporal Features (5 features)

#### Time-Based Features:
- **hour**: Transaction hour (0-23, int8)
- **dow**: Day of week (0-6, int8)
- **is_weekend**: Binary flag for Saturday/Sunday (int8)
- **is_night**: Binary flag for 0-5 AM (int8)
- **month**: Month of year (1-12, int8)

### 2. Account & PIN Age Features (2 features)

- **acct_age_days**: Days since account opening (int32)
- **pin_age_years**: Years since last PIN change (int16)

### 3. Amount Features (3 features)

- **amount_log**: Log transformation of amount (log1p, float32)
- **amount_high**: Binary flag for amount > 500 (int8)
- **amt_per_day**: Amount per account age day (float32)

### 4. Geographic Features (1 feature)

- **dist_home_km**: Haversine distance from user's home to merchant (vectorized, float32)
  - **Home location**: Median latitude/longitude per card_id
  - **Calculation**: Vectorized Haversine formula (10x faster than iterative)
  - **Fallback**: If no home location, uses transaction location

### 5. Target Encoding Features (6 features)

Target encoding calculates fraud rate for each category:

- **te_card_id**: Fraud rate per card (float32)
- **te_client_id**: Fraud rate per client (float32)
- **te_merchant_state**: Fraud rate per state (float32)
- **te_merchant_id**: Fraud rate per merchant (float32) 
- **te_mcc**: Fraud rate per MCC code (float32) 
- **te_use_chip**: Fraud rate per chip usage (float32) - dropped in final model

**Handling**: Global mean used for unseen categories

### 6. Frequency Encoding Features (4 features)

Frequency encoding counts occurrences across train + test:

- **freq_card_id**: Transaction count per card (int32)
- **freq_client_id**: Transaction count per client (int32)
- **freq_merchant_state**: Transaction count per state (int32)
- **freq_merchant_id**: Transaction count per merchant (int32) 

### 7. Interaction Features (8 features)

Risk-aware interactions combining multiple signals:

- **amt_x_dist**: Amount × distance (float32)
- **amt_per_card**: Amount / card frequency (float32)
- **log_amt_per_hour**: Log amount / hour (float32)
- **dist_per_card**: Distance / card frequency (float32)
- **amt_x_merchant_risk**: Amount × merchant fraud rate
- **dist_x_merchant_risk**: Distance × merchant fraud rate 
- **mcc_risk_x_amount**: MCC fraud rate × amount 
- **amt_per_merchant**: Amount / merchant frequency 

### 8. Raw Features (18 features)

- **Numerical**: client_id, card_id, amount, merchant_id, mcc, zip, num_cards_issued, credit_limit, current_age, retirement_age, birth_year, birth_month, latitude, longitude, per_capita_income, yearly_income, total_debt, credit_score, num_credit_cards
- **Categorical**: merchant_state, errors, card_brand, card_type, card_on_dark_web, gender, mcc_description

### Feature Engineering Insights

#### Key Features (Highest Impact):
1. **te_merchant_id**: Merchant fraud rate (strongest signal)
2. **te_mcc**: MCC fraud rate (merchant category risk)
3. **freq_merchant_id**: Merchant frequency (behavioral pattern)
4. **Merchant risk interactions**: Amount × merchant risk, distance × merchant risk
5. **dist_home_km**: Geographic distance (fraud often far from home)

#### Dropped Features (Data Drift):
- **use_chip**: 100% missing in test set
- **has_chip**: 100% missing in test set
- **te_use_chip**: Derived from use_chip

#### Final Feature Count:
- **Total**: 47 features (all numeric after encoding)
- **Training**: 6,240,474 rows × 47 features
- **Test**: 2,674,489 rows × 47 features

<br>

___

<br>

## Model Development

### 1. Model Selection

#### Chosen Model: XGBoost with GPU Acceleration

**Rationale**:
- **Excellent performance**: Gradient boosting excels on tabular data
- **GPU acceleration**: Fast training on large datasets (~14 minutes for 5-fold CV)
- **Class imbalance handling**: Built-in `scale_pos_weight` parameter
- **Feature importance**: Interpretable feature contributions
- **Regularization**: L1/L2 regularization prevents overfitting

### 2. Class Imbalance Handling

#### Scale Positive Weight:
```python
fraud_rate = 0.001495  # 0.15% fraud rate
scale_pos_weight = (1 - fraud_rate) / fraud_rate  # ~668
```

This ensures the model treats fraud cases as 668x more important than legitimate cases during training.

### 3. Hyperparameters

#### Final XGBoost Parameters:
```python
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'gpu_hist',          # GPU acceleration
    'gpu_id': 0,
    'predictor': 'gpu_predictor',
    'random_state': 42,
    'learning_rate': 0.05,              # Conservative learning rate
    'max_depth': 9,                     # Deep trees for complex patterns
    'subsample': 0.8,                   # 80% sampling for regularization
    'colsample_bytree': 0.8,            # 80% feature sampling
    'reg_alpha': 0.2,                   # L1 regularization
    'reg_lambda': 1.5,                  # L2 regularization (strong)
    'min_child_weight': 3,              # Minimum samples in leaf
    'scale_pos_weight': 668,            # Handle class imbalance
    'n_estimators': 3000,               # Large number (early stopping)
    'early_stopping_rounds': 75         # Prevent overfitting
}
```

#### Key Design Decisions:
- **Deep trees (depth=9)**: Capture complex fraud patterns
- **Strong regularization**: reg_lambda=1.5 prevents overfitting on imbalanced data
- **Early stopping**: Stops training after 75 rounds without improvement
- **GPU acceleration**: 10-50x faster than CPU

### 4. Validation Strategy

#### 5-Fold Stratified K-Fold Cross-Validation:
```python
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
```

**Benefits**:
- **Stratified**: Maintains fraud rate in each fold (~0.15%)
- **Robust evaluation**: 5-fold CV provides stable performance estimates
- **Out-of-fold predictions**: Used for threshold optimization

### 5. Threshold Optimization

#### 2-Phase Threshold Tuning:

**Phase 1: Coarse Search**
- Search range: 0.3 to 0.95
- Step size: 30 points
- Finds approximate optimal threshold

**Phase 2: Fine Search**
- Search range: Best threshold ± 0.05
- Step size: 50 points
- Refines optimal threshold

**Optimization Metric**: Cohen Kappa Score (competition metric)

**Final Threshold**: 0.717 (average across 5 folds)

### 6. Training Process

#### Cross-Validation Loop:
1. **Split data**: 5-fold stratified split
2. **Train model**: XGBoost on training fold
3. **Validate**: Predict on validation fold
4. **Optimize threshold**: Find best threshold for Kappa on validation fold
5. **Test prediction**: Predict on test set
6. **Average**: Average test predictions across 5 folds

#### Training Time:
- **Per fold**: ~2-3 minutes on GPU
- **Total**: ~14 minutes for 5-fold CV
- **Early stopping**: Typically stops around 1,500-1,600 iterations

### 7. Results

#### Cross-Validation Performance:
- **Mean CV Kappa**: 0.89627
- **OOF Kappa**: 0.89522
- **Best threshold**: 0.717

#### Fold Performance:
- Fold 1: Kappa = 0.89565 @ threshold = 0.844
- Fold 2: Kappa = 0.89637 @ threshold = 0.698
- Fold 3: Kappa = 0.89442 @ threshold = 0.711
- Fold 4: Kappa = 0.89603 @ threshold = 0.658
- Fold 5: Kappa = 0.89887 @ threshold = 0.671

#### Final Predictions:
- **Fraud predictions**: 3,256 transactions (0.12% of test set)
- **Legitimate predictions**: 2,671,233 transactions (99.88% of test set)

<br>

___

<br>

## Code Walkthrough

### Step 1: Data Preprocessing

#### 1.1 Load and Index Data
```python
train_df = pd.read_parquet("/kaggle/input/nwu-datathon/merged_train_dataset.parquet")
test_df  = pd.read_parquet("/kaggle/input/nwu-datathon/test_merged_data.parquet")

# Set transaction_id as index
train_df = train_df.set_index('transaction_id', verify_integrity=True)
test_df  = test_df.set_index('transaction_id',  verify_integrity=True)
```

#### 1.2 Drop PII and Clean Data
```python
# Drop sensitive columns
drop_cols = ['card_number', 'cvv', 'address', 'merchant_city']
train_df = train_df.drop(columns=drop_cols, errors='ignore')

# Convert fraud labels
train_df['fraud'] = train_df['fraud'].map({'Yes':1, 'No':0}).astype('int8')

# Clean money columns
money_cols = ['amount','credit_limit','per_capita_income','yearly_income','total_debt']
for col in money_cols:
    train_df[col] = pd.to_numeric(
        train_df[col].astype(str).str.replace(r'[\$,]', '', regex=True), 
        errors='coerce'
    )
```

#### 1.3 Parse Dates and Extract Features
```python
# Parse dates
for col in ['date','acct_open_date','expires','year_pin_last_changed']:
    train_df[col] = pd.to_datetime(train_df[col], errors='coerce')

# Extract PIN year
train_df['pin_year'] = train_df['year_pin_last_changed'].dt.year.fillna(-999).astype('int16')
```

### Step 2: Feature Engineering

#### 2.1 Temporal Features
```python
for df in [train, test]:
    df['hour']       = df['date'].dt.hour.astype('int8')
    df['dow']        = df['date'].dt.dayofweek.astype('int8')
    df['is_weekend'] = (df['dow'] >= 5).astype('int8')
    df['is_night']   = df['hour'].between(0, 5).astype('int8')
    df['month']      = df['date'].dt.month.astype('int8')
```

#### 2.2 Geographic Features (Vectorized)
```python
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return (2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).astype('float32')

# Calculate home location (median per card)
home = train.groupby('card_id')[['latitude','longitude']].median()
home.columns = ['home_lat','home_lon']

# Calculate distance from home
train = train.join(home, on='card_id', how='left')
train['dist_home_km'] = haversine_vectorized(
    train['latitude'].values, train['longitude'].values,
    train['home_lat'].values, train['home_lon'].values
)
```

#### 2.3 Target Encoding
```python
global_mean = train['fraud'].mean()
te_cols = ['card_id', 'client_id', 'merchant_state', 'merchant_id', 'mcc']

for col in te_cols:
    means = train.groupby(col)['fraud'].mean().astype('float32')
    train[f'te_{col}'] = train[col].map(means).fillna(global_mean).astype('float32')
    test[f'te_{col}']  = test[col].map(means).fillna(global_mean).astype('float32')
```

#### 2.4 Frequency Encoding
```python
freq_cols = ['card_id', 'client_id', 'merchant_state', 'merchant_id']

for col in freq_cols:
    all_col = pd.concat([train[col], test[col]], axis=0)
    cnt = all_col.value_counts()
    train[f'freq_{col}'] = train[col].map(cnt).astype('int32')
    test[f'freq_{col}']  = test[col].map(cnt).astype('int32')
```

#### 2.5 Interaction Features
```python
for df in [train, test]:
    df['amt_x_merchant_risk'] = (df['amount'] * df['te_merchant_id']).astype('float32')
    df['dist_x_merchant_risk'] = (df['dist_home_km'] * df['te_merchant_id']).astype('float32')
    df['mcc_risk_x_amount'] = (df['te_mcc'] * df['amount']).astype('float32')
    df['amt_per_merchant'] = (df['amount'] / (df['freq_merchant_id'] + 1)).astype('float32')
```

### Step 3: Model Training

#### 3.1 Prepare Data
```python
# Drop drift columns (100% missing in test)
drift_drop_cols = ['use_chip', 'has_chip', 'te_use_chip']
train.drop(columns=drift_drop_cols, inplace=True)
test.drop(columns=drift_drop_cols, inplace=True)

# Encode categoricals
cat_cols = ['merchant_state', 'errors', 'card_brand', 'card_type', 
            'card_on_dark_web', 'gender', 'mcc_description']
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col].astype(str), test[col].astype(str)])
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str)).astype('int32')
    test[col]  = le.transform(test[col].astype(str)).astype('int32')

# Prepare X, y
X = train.drop(columns=['fraud'])
y = train['fraud']
X_test = test.copy()
```

#### 3.2 XGBoost Training
```python
# Calculate scale_pos_weight
fraud_rate = y.mean()
scale_pos_weight = (1 - fraud_rate) / fraud_rate

# XGBoost parameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'predictor': 'gpu_predictor',
    'random_state': 42,
    'learning_rate': 0.05,
    'max_depth': 9,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'min_child_weight': 3,
    'scale_pos_weight': scale_pos_weight,
    'n_estimators': 3000,
    'early_stopping_rounds': 75
}

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
    model = XGBClassifier(**xgb_params)
    model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], 
              early_stopping_rounds=75, verbose=100)
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / 5
```

#### 3.3 Threshold Optimization
```python
# 2-phase threshold optimization
# Phase 1: Coarse search
thresholds_coarse = np.linspace(0.3, 0.95, 30)
best_kappa = -1
best_thresh = 0.5
for thresh in thresholds_coarse:
    pred = (oof_preds[val_idx] >= thresh).astype(int)
    kappa = cohen_kappa_score(y_val, pred)
    if kappa > best_kappa:
        best_kappa = kappa
        best_thresh = thresh

# Phase 2: Fine search
fine_start = max(0.3, best_thresh - 0.05)
fine_end = min(0.95, best_thresh + 0.05)
thresholds_fine = np.linspace(fine_start, fine_end, 50)
for thresh in thresholds_fine:
    pred = (oof_preds[val_idx] >= thresh).astype(int)
    kappa = cohen_kappa_score(y_val, pred)
    if kappa > best_kappa:
        best_kappa = kappa
        best_thresh = thresh
```

#### 3.4 Generate Submission
```python
final_threshold = np.mean(best_thresholds)  # 0.717
submission = pd.DataFrame({
    'transaction_id': test.index,
    'fraud': np.where(test_preds >= final_threshold, 'Yes', 'No')
})
submission.to_csv('submission.csv', index=False)
```

<br>

___

<br>

## Results & Performance

### Competition Results

- **Final Ranking**: #6 among 40 teams
- **Evaluation Metric**: Cohen Kappa Score
- **Final Score**: 0.90425
- **CV Performance**: Mean CV Kappa = 0.89627, OOF Kappa = 0.89522
- **Training Time**: ~14 minutes on GPU (5-fold CV)
- **Predictions**: 3,256 fraud cases (0.12% of test set) 



<br>

___

<br>

## Key Insights

### 1. **Target Encoding is the Game Changer**
- **Lesson**: Target encoding for high-cardinality categoricals (merchant_id, mcc, card_id) provided the strongest fraud signals
- **Impact**: `te_merchant_id` and `te_mcc` were among the top features
- **Key Insight**: Fraud patterns are highly merchant and category-specific
- **Takeaway**: Always use target encoding for high-cardinality categoricals in fraud detection

### 2. **Data Drift Detection is Critical**
- **Lesson**: Test set had 100% missing values for `use_chip` and `has_chip` (train had values)
- **Solution**: Identified and dropped drift columns before modeling
- **Impact**: Prevented model from learning on features unavailable at test time
- **Takeaway**: Always compare train/test distributions for categorical columns

### 3. **Threshold Optimization Matters for Kappa**
- **Lesson**: Default threshold (0.5) is suboptimal for Cohen Kappa score
- **Solution**: 2-phase threshold optimization (coarse → fine) on validation set
- **Impact**: Optimal threshold (0.717) significantly improved Kappa score
- **Key Insight**: Kappa rewards balanced predictions, not just accuracy

### 4. **Vectorized Operations are Essential**
- **Lesson**: Vectorized Haversine distance calculation is 10x faster than iterative
- **Solution**: Used NumPy vectorized operations for geographic features
- **Impact**: Reduced feature engineering time from hours to minutes
- **Takeaway**: Always vectorize operations when possible

### 5. **Merchant Risk Interactions are Powerful**
- **Lesson**: Interactions between amount/distance and merchant fraud rate are highly predictive
- **Key Features**: `amt_x_merchant_risk`, `dist_x_merchant_risk`, `mcc_risk_x_amount`
- **Impact**: These interactions captured complex fraud patterns
- **Takeaway**: Create risk-aware interactions for fraud detection

### 6. **GPU Acceleration is a Must for Large Datasets**
- **Lesson**: XGBoost GPU training is 10-50x faster than CPU
- **Impact**: 5-fold CV completed in ~14 minutes vs hours on CPU
- **Key Insight**: GPU enables rapid experimentation and hyperparameter tuning
- **Takeaway**: Always use GPU when available for large-scale machine learning

### 7. **Scale Pos Weight Handles Extreme Imbalance**
- **Lesson**: `scale_pos_weight = 668` (fraud rate 0.15%) effectively handles class imbalance
- **Impact**: Model learned to detect fraud without oversampling/undersampling
- **Key Insight**: XGBoost's scale_pos_weight is simpler and more effective than SMOTE
- **Takeaway**: Use scale_pos_weight for extreme class imbalance in XGBoost

### 8. **Regularization Prevents Overfitting on Imbalanced Data**
- **Lesson**: Strong L2 regularization (reg_lambda=1.5) prevents overfitting on rare fraud cases
- **Impact**: Model generalizes well despite extreme class imbalance
- **Key Insight**: Imbalanced data requires stronger regularization
- **Takeaway**: Increase regularization strength for imbalanced datasets

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
pip install pandas numpy scikit-learn xgboost
```

**Note**: For GPU acceleration, ensure CUDA is installed and XGBoost is compiled with GPU support.

### 2. **Load Data**

```python
import pandas as pd

# Load pre-merged Parquet files (or merge using merge_all.py)
train = pd.read_parquet('merged_train_dataset.parquet')
test = pd.read_parquet('test_merged_data.parquet')

# Set transaction_id as index
train = train.set_index('transaction_id', verify_integrity=True)
test = test.set_index('transaction_id', verify_integrity=True)
```

### 3. **Data Preprocessing**

See the **Code Walkthrough** section above for complete preprocessing steps:
- Drop PII columns
- Clean money columns (remove $ and commas)
- Parse dates
- Handle missing values
- Optimize data types

### 4. **Feature Engineering**

See the **Code Walkthrough** section above for complete feature engineering:
- Temporal features (hour, dow, is_weekend, is_night, month)
- Account age features (acct_age_days, pin_age_years)
- Amount features (amount_log, amount_high, amt_per_day)
- Geographic features (dist_home_km - vectorized Haversine)
- Target encoding (te_merchant_id, te_mcc, te_card_id, etc.)
- Frequency encoding (freq_merchant_id, freq_card_id, etc.)
- Interaction features (amt_x_merchant_risk, dist_x_merchant_risk, etc.)

### 5. **Model Training**

```python
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Prepare data (drop drift columns, encode categoricals)
# See Code Walkthrough for details

# Calculate scale_pos_weight
fraud_rate = train['fraud'].mean()
scale_pos_weight = (1 - fraud_rate) / fraud_rate

# XGBoost parameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'gpu_hist',  # Use 'hist' for CPU
    'gpu_id': 0,
    'predictor': 'gpu_predictor',
    'random_state': 42,
    'learning_rate': 0.05,
    'max_depth': 9,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'min_child_weight': 3,
    'scale_pos_weight': scale_pos_weight,
    'n_estimators': 3000,
    'early_stopping_rounds': 75
}

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
    
    model = XGBClassifier(**xgb_params)
    model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], 
              early_stopping_rounds=75, verbose=100)
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / 5
```

### 6. **Threshold Optimization**

```python
from sklearn.metrics import cohen_kappa_score

# 2-phase threshold optimization
# Phase 1: Coarse search
thresholds_coarse = np.linspace(0.3, 0.95, 30)
best_kappa = -1
best_thresh = 0.5
for thresh in thresholds_coarse:
    pred = (oof_preds[val_idx] >= thresh).astype(int)
    kappa = cohen_kappa_score(y_val, pred)
    if kappa > best_kappa:
        best_kappa = kappa
        best_thresh = thresh

# Phase 2: Fine search
fine_start = max(0.3, best_thresh - 0.05)
fine_end = min(0.95, best_thresh + 0.05)
thresholds_fine = np.linspace(fine_start, fine_end, 50)
for thresh in thresholds_fine:
    pred = (oof_preds[val_idx] >= thresh).astype(int)
    kappa = cohen_kappa_score(y_val, pred)
    if kappa > best_kappa:
        best_kappa = kappa
        best_thresh = thresh

final_threshold = np.mean(best_thresholds)  # Average across folds
```

### 7. **Generate Submission**

```python
# Create submission
submission = pd.DataFrame({
    'transaction_id': test.index,
    'fraud': np.where(test_preds >= final_threshold, 'Yes', 'No')
})

submission.to_csv('submission.csv', index=False)
```

### 8. **Key Notes**

- **GPU Acceleration**: Use `tree_method='gpu_hist'` for GPU, `'hist'` for CPU
- **Data Drift**: Always check for columns with 100% missing in test set
- **Threshold Optimization**: 2-phase optimization is critical for Cohen Kappa
- **Target Encoding**: Must be done on training set only, then mapped to test
- **Frequency Encoding**: Can be done on combined train+test for better estimates

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
├── my-notebook.ipynb                    # Main competition notebook
│
└── README.md                            # This file
```

### Notebook Structure

The `my-notebook.ipynb` contains the complete pipeline:
1. **Data Loading**: Load and inspect merged Parquet files
2. **Data Preprocessing**: Clean, index, and prepare data
3. **Feature Engineering**: Create 47 features
4. **Model Training**: XGBoost with 5-fold CV
5. **Threshold Optimization**: 2-phase optimization for Kappa
6. **Submission Generation**: Create submission file

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






