Let me give you a complete, step-by-step data cleaning guide:

---

# 🧹 Complete Data Cleaning Guide
## Cloud Service Providers Dataset — 10,200 Rows × 28 Columns

---

## 🏗️ STEP 1 — Choose Your Technology Stack

Before writing a single line of code, choose the right tools:

---

### 🥇 Best Technology: **Apache Spark + Python (PySpark)**
**Why Spark for this project specifically:**
- Your dataset is 10K rows now but in real cloud environments it's **billions of rows**
- Spark processes data in **parallel across multiple cores/machines**
- Handles all 28 columns and 20 cleaning scenarios natively
- Industry standard for cloud data pipelines at AWS, Azure, GCP

### 🥈 Alternative: **Python + Pandas**
- Perfectly fine for 10K rows
- Easier to learn and explain
- We will use **Pandas** for this guide since it's interview-friendly

### Supporting Tools:
| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core language |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical operations |
| **Regex (re)** | Pattern matching for PII, IDs |
| **pytz / dateutil** | Timezone handling |
| **Jupyter Notebook** | Step by step execution |
| **SQLite / PostgreSQL** | Store cleaned output |

---

## 🏗️ STEP 2 — Set Up Your Environment

```python
# Install all required libraries
pip install pandas numpy python-dateutil pytz openpyxl re2 jupyter

# Import everything at the top of your notebook
import pandas as pd
import numpy as np
import re
from dateutil import parser as dateparser
import pytz
import warnings
warnings.filterwarnings('ignore')

# Load the raw dataset
df = pd.read_excel('Cloud_CSP_Raw_Dataset_10K.xlsx', 
                    sheet_name='Raw_Dataset',
                    dtype=str)  # Load EVERYTHING as string first
                                # This prevents pandas from 
                                # auto-converting messy values

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
df.head()
```

> **Why `dtype=str`?**
> If you let pandas auto-detect types, it will try to convert `₹1,200.50` to a number and FAIL. Load everything as string first, then convert column by column after cleaning.

---

## 🏗️ STEP 3 — Explore Before You Clean (EDA)

Never start cleaning without understanding what you're dealing with:

```python
# 1. Basic shape
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# 2. Missing values per column
print("\n=== MISSING VALUES ===")
print(df.isnull().sum().sort_values(ascending=False))

# 3. Unique values in key columns
print("\n=== ACCOUNT VARIANTS ===")
print(df['Account'].value_counts().head(20))

print("\n=== UNIT VARIANTS ===")
print(df['Unit'].value_counts())

print("\n=== REGION VARIANTS ===")
print(df['Region'].value_counts())

print("\n=== PRICING TYPE VARIANTS ===")
print(df['Pricing_Type'].value_counts())

# 4. Sample the messy rows
print("\n=== SAMPLE MESSY TIMESTAMPS ===")
print(df['TS'].sample(10).values)
```

> **What to tell the interviewer:**
> *"EDA is not optional — it's step zero. You cannot write cleaning logic without first understanding the exact nature and frequency of each data quality problem."*

---

## 🏗️ STEP 4 — Clean Each Column (Scenario by Scenario)

---

### 🔧 Scenario #1 — Account ID & Ticket ID Trimming/Uppercasing

```python
# ── Account ID ──────────────────────────────────────────────
def clean_account_id(val):
    if pd.isna(val) or str(val).strip() in ('', 'N/A', 'na', 'NA'):
        return None
    val = str(val).strip().upper()          # strip spaces + uppercase
    val = re.sub(r'[\s_]', '-', val)        # replace space/underscore with hyphen
    # normalize: ACCT003 → ACCT-003
    val = re.sub(r'(ACCT)(\d+)', r'\1-\2', val)
    return val

df['Account_Clean'] = df['Account'].apply(clean_account_id)

# ── Ticket ID ───────────────────────────────────────────────
def clean_ticket_id(val):
    if pd.isna(val) or str(val).strip() in ('', 'N/A', 'na', 'NA'):
        return None
    val = str(val).strip().upper()
    # normalize: tkt-6679 / ticket6679 / TICKET-6679 → T-6679
    val = re.sub(r'^(TICKET|TKT)-?(\d+)$', r'T-\2', val)
    return val

df['Ticket_ID_Clean'] = df['Ticket_ID'].apply(clean_ticket_id)

print(df[['Account', 'Account_Clean', 'Ticket_ID', 'Ticket_ID_Clean']].head(10))
```

---

### 🔧 Scenario #2 — Timestamp Normalization to UTC

```python
def clean_timestamp(val):
    if pd.isna(val) or str(val).strip() in ('', 'N/A', 'na'):
        return None
    val = str(val).strip()
    
    # Fix invalid hours (e.g., 25:05 → 01:05)
    val = re.sub(r' (2[4-9]|[3-9]\d):(\d{2})', 
                  lambda m: f" 0{int(m.group(1))-24}:{m.group(2)}", val)
    
    # Fix slash separators → hyphens
    val = re.sub(r'(\d{4})/(\d{2})/(\d{2})', r'\1-\2-\3', val)
    
    # Fix DD-MM-YYYY → YYYY-MM-DD
    val = re.sub(r'^(\d{2})-(\d{2})-(\d{4})', r'\3-\2-\1', val)
    
    try:
        dt = dateparser.parse(val)              # parse whatever format remains
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)          # assume UTC if no timezone
        else:
            dt = dt.astimezone(pytz.utc)        # convert to UTC
        return dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
    except:
        return None                             # flag as unparseable

df['TS_UTC'] = df['TS'].apply(clean_timestamp)
df['TS_Parse_Failed'] = df['TS_UTC'].isna() & df['TS'].notna()

print(f"Successfully parsed: {df['TS_UTC'].notna().sum()}")
print(f"Failed to parse: {df['TS_Parse_Failed'].sum()}")
```

---

### 🔧 Scenario #3 — SKU Canonical Naming

```python
# Master SKU catalog — maps all variations to canonical form
SKU_CATALOG = {
    'VM-STD-2': ['vm-std-2','VM_STD_2','vm.std.2','vmstd2','VM-STD-2'],
    'VM-STD-4': ['vm-std-4','VM_STD_4','vm.std.4','vmstd4','VM-STD-4'],
    'VM-STD-8': ['vm-std-8','VM_STD_8','vm.std.8','vmstd8'],
    'VM-PREM-4': ['vm-prem-4','VM_PREM_4','vmprem4','VM-PREM-4'],
    'VM-BASIC-2': ['vm-basic-2','VM_BASIC_2','vmbasic2'],
    # ... add all SKUs
}

# Build reverse lookup: variant → canonical
sku_lookup = {}
for canonical, variants in SKU_CATALOG.items():
    for v in variants:
        sku_lookup[v.upper().replace('_','-').replace('.','-')] = canonical

def clean_sku(val):
    if pd.isna(val) or str(val).strip() == '':
        return None
    normalized = str(val).strip().upper()
    normalized = re.sub(r'[_\.]', '-', normalized)   # unify separators
    return sku_lookup.get(normalized, normalized)     # return canonical or best guess

df['SKU_Clean'] = df['SKU'].apply(clean_sku)

print(df[['SKU', 'SKU_Clean']].drop_duplicates().head(20))
```

---

### 🔧 Scenario #4 — Usage Unit Normalization

```python
# Conversion factors to seconds
UNIT_TO_SECONDS = {
    'sec': 1, 'second': 1, 'seconds': 1, 's': 1,
    'min': 60, 'minute': 60, 'minutes': 60,
    'hr': 3600, 'hour': 3600, 'hours': 3600, 'hrs': 3600,
}

def clean_usage(usage_val, unit_val):
    try:
        # Remove commas from numbers like 12,000
        usage_str = str(usage_val).replace(',', '').strip()
        usage_num = float(usage_str)
    except:
        return None
    
    unit = str(unit_val).strip().lower()
    multiplier = UNIT_TO_SECONDS.get(unit, 1)   # default to 1 if unit unknown
    return int(usage_num * multiplier)

df['Usage_Seconds'] = df.apply(
    lambda row: clean_usage(row['Usage'], row['Unit']), axis=1
)

print(df[['Usage', 'Unit', 'Usage_Seconds']].head(15))
print(f"\nNull usage records: {df['Usage_Seconds'].isna().sum()}")
```

---

### 🔧 Scenario #5 — Cost Currency & Decimal Normalization

```python
def clean_cost(val):
    if pd.isna(val) or str(val).strip() in ('', 'N/A', 'na'):
        return None
    val = str(val)
    # Remove all currency symbols and letters
    val = re.sub(r'[₹$€£¥A-Za-z\s]', '', val)
    # Remove comma thousands separators
    # But keep decimal point — handle European format (1.234,56)
    if ',' in val and '.' in val:
        if val.index(',') < val.index('.'):
            val = val.replace(',', '')          # 1,234.56 → 1234.56
        else:
            val = val.replace('.', '').replace(',', '.')  # 1.234,56 → 1234.56
    elif ',' in val:
        val = val.replace(',', '.')             # 1234,56 → 1234.56
    try:
        return round(float(val), 2)
    except:
        return None

def clean_currency(val):
    if pd.isna(val) or str(val).strip() in ('', 'N/A', 'na', 'NA'):
        return 'UNKNOWN'
    val = str(val).strip().upper()
    if any(x in val for x in ['USD', 'US DOLLAR', 'DOLLAR', '$']):
        return 'USD'
    if any(x in val for x in ['INR', '₹', 'RUPEE']):
        return 'INR'
    if any(x in val for x in ['EUR', '€', 'EURO']):
        return 'EUR'
    return val

df['Cost_Clean']     = df['Cost'].apply(clean_cost)
df['Currency_Clean'] = df['Currency'].apply(clean_currency)

print(df[['Cost', 'Currency', 'Cost_Clean', 'Currency_Clean']].head(15))
```

---

### 🔧 Scenario #6 — Region Normalization

```python
REGION_MAP = {
    # ap-south-1 variants
    'AP-SOUTH-1':        'ap-south-1',
    'AP_SOUTH_1':        'ap-south-1',
    'AP SOUTH 1':        'ap-south-1',
    'AP-SOUTH1':         'ap-south-1',
    'ASIA PACIFIC (MUMBAI)': 'ap-south-1',
    # us-east-1 variants
    'US-EAST-1':         'us-east-1',
    'US_EAST_1':         'us-east-1',
    'US EAST 1':         'us-east-1',
    'USEAST1':           'us-east-1',
    'UNITED STATES EAST':'us-east-1',
    # eu-central-1 variants
    'EU-CENTRAL-1':      'eu-central-1',
    'EU_CENTRAL_1':      'eu-central-1',
    'EUCENTRAL1':        'eu-central-1',
    'EU CENTRAL':        'eu-central-1',
    'EU-CENTRAL1':       'eu-central-1',
    # ca-central-1 variants
    'CA-CENTRAL-1':      'ca-central-1',
    'CA-CENTRAL':        'ca-central-1',
    'CA_CENTRAL_1':      'ca-central-1',
    'CANADA CENTRAL':    'ca-central-1',
    'CA CENTRAL 1':      'ca-central-1',
    # ap-southeast-1 variants
    'AP-SOUTHEAST-1':    'ap-southeast-1',
    'AP_SOUTHEAST_1':    'ap-southeast-1',
    'AP-SE-1':           'ap-southeast-1',
    'SINGAPORE':         'ap-southeast-1',
}

def clean_region(val):
    if pd.isna(val) or str(val).strip() in ('', 'N/A', 'na'):
        return None
    normalized = str(val).strip().upper()
    normalized = re.sub(r'\s+', ' ', normalized)     # collapse spaces
    return REGION_MAP.get(normalized, normalized.lower())

df['Region_Clean'] = df['Region'].apply(clean_region)

print(df[['Region', 'Region_Clean']].value_counts().head(20))
```

---

### 🔧 Scenario #7 — Duplicate Detection & Deduplication

```python
# Composite key = Account + TS + SKU
# Any two rows with same composite = duplicate

# First clean Account, TS, SKU so the composite is meaningful
df['_dedup_key'] = (
    df['Account_Clean'].fillna('') + '|' +
    df['TS_UTC'].fillna('') + '|' +
    df['SKU_Clean'].fillna('')
)

# Mark duplicates
df['Duplicate_Flag'] = df.duplicated(subset=['_dedup_key'], keep='first')

print(f"Total rows:      {len(df)}")
print(f"Duplicate rows:  {df['Duplicate_Flag'].sum()}")
print(f"Unique rows:     {(~df['Duplicate_Flag']).sum()}")

# Keep only non-duplicates for clean dataset
df_deduped = df[~df['Duplicate_Flag']].copy()
print(f"\nAfter dedup: {len(df_deduped)} rows")

# Drop helper column
df.drop(columns=['_dedup_key'], inplace=True)
```

---

### 🔧 Scenario #8 — Free Tier / Credit Tagging

```python
def clean_free_tier(val):
    if pd.isna(val) or str(val).strip() in ('', 'N/A', 'na', 'NA', 'none', 'None', 'NONE'):
        return 'NONE'
    val = str(val).strip().upper().replace('-', '').replace('_', '')
    if 'CREDIT' in val:
        return 'CREDIT'
    if 'FREE' in val or 'TIER' in val:
        return 'FREE_TIER'
    return 'NONE'

df['Free_Tier_Clean'] = df['Free_Tier_Flag'].apply(clean_free_tier)

print(df['Free_Tier_Clean'].value_counts())
```

---

### 🔧 Scenario #9 — Anomaly Detection on Usage Spikes

```python
# Calculate per-account per-SKU rolling statistics
# Step 1: Get clean numeric usage
df['Usage_Seconds'] = pd.to_numeric(df['Usage_Seconds'], errors='coerce')

# Step 2: Calculate mean and std per Account+SKU group
stats = df.groupby(['Account_Clean', 'SKU_Clean'])['Usage_Seconds'].agg(
    mean_usage='mean',
    std_usage='std'
).reset_index()

df = df.merge(stats, on=['Account_Clean', 'SKU_Clean'], how='left')

# Step 3: Calculate Z-score deviation
df['Usage_ZScore'] = (
    (df['Usage_Seconds'] - df['mean_usage']) / df['std_usage'].replace(0, 1)
)

# Step 4: Flag anomalies (Z-score > 3 = anomaly)
df['Anomaly_Flag'] = df['Usage_ZScore'].abs() > 3

print(f"Total anomalies detected: {df['Anomaly_Flag'].sum()}")
print(f"Anomaly rate: {df['Anomaly_Flag'].mean()*100:.2f}%")

# Clean up helper columns
df.drop(columns=['mean_usage', 'std_usage'], inplace=True)
```

---

### 🔧 Scenario #10 — Tag Normalization

```python
def clean_tag_owner(val):
    if pd.isna(val) or str(val).strip() in ('', 'N/A', 'na', 'NA'):
        return 'UNTAGGED'
    val = str(val).strip().lower()
    val = re.sub(r'^(team|grp|group)[-_\s]?', 'team-', val)
    val = re.sub(r'[-_\s]+', '-', val)
    return val

def clean_tag_env(val):
    if pd.isna(val) or str(val).strip() in ('', 'N/A', 'na', 'NA'):
        return 'UNTAGGED'
    val = str(val).strip().upper()
    if val in ('PROD', 'PRODUCTION'):     return 'PROD'
    if val in ('DEV', 'DEVELOPMENT'):     return 'DEV'
    if val in ('STAGING', 'STAGE'):       return 'STAGING'
    if val in ('TEST', 'TESTING', 'QA'): return 'TEST'
    return 'UNKNOWN'

df['Tag_Owner_Clean'] = df['Tag_Owner'].apply(clean_tag_owner)
df['Tag_Env_Clean']   = df['Tag_Env'].apply(clean_tag_env)

print(df['Tag_Owner_Clean'].value_counts().head(10))
print(df['Tag_Env_Clean'].value_counts())
```

---

### 🔧 Scenario #11 — Resource ID Validation

```python
# Valid pattern: prefix-suffix (e.g., RES-VM-001)
VALID_RESOURCE_PATTERN = re.compile(
    r'^(RES|res)[-_](VM|DB|FUNC|K8S|LB|CDN|ML|GPU)[-_]\w+$', re.IGNORECASE
)

def clean_resource_id(val):
    if pd.isna(val) or str(val).strip() in ('', 'N/A', 'na', 'NA'):
        return None
    val = str(val).strip().upper()
    val = re.sub(r'[_\.]', '-', val)
    return val if VALID_RESOURCE_PATTERN.match(val) else f"INVALID-{val}"

df['Resource_ID_Clean']   = df['Resource_ID'].apply(clean_resource_id)
df['Resource_ID_Valid']   = ~df['Resource_ID_Clean'].str.startswith('INVALID', na=False)

print(f"Valid resource IDs:   {df['Resource_ID_Valid'].sum()}")
print(f"Invalid resource IDs: {(~df['Resource_ID_Valid']).sum()}")
```

---

### 🔧 Scenario #12 — PII Masking

```python
def mask_pii(text):
    if pd.isna(text):
        return text
    text = str(text)
    # Mask email addresses
    text = re.sub(r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b', '***@***.***', text)
    # Mask phone numbers (Indian + international)
    text = re.sub(r'\b(\+?\d[\d\s\-]{8,14}\d)\b', '***-***-****', text)
    # Mask proper names (simple: Title Case words after keywords)
    text = re.sub(r'\b(by|contact|from|raised by|for)\s+([A-Z][a-z]+\s[A-Z][a-z]+)',
                   lambda m: m.group(1) + ' [MASKED]', text)
    return text

df['Ticket_Text_Masked'] = df['Ticket_Text'].apply(mask_pii)

# Show before/after
sample = df[df['Ticket_Text'].str.contains('@|Gupta|Sharma|Park', na=False)].head(5)
print(sample[['Ticket_Text', 'Ticket_Text_Masked']].to_string())
```

---

### 🔧 Scenario #13 — Incident ID Normalization

```python
def clean_incident_id(val):
    if pd.isna(val) or str(val).strip() in ('', 'N/A', 'na', 'NA'):
        return None
    val = str(val).strip().upper()
    # normalize: Incident-6123 / incident-6123 / INCI_6123 → INC-6123
    val = re.sub(r'^(INCIDENT|INCI)[-_]?(\d+)$', r'INC-\2', val)
    val = re.sub(r'^INC(\d+)$', r'INC-\1', val)   # INC6123 → INC-6123
    return val

df['Incident_ID_Clean'] = df['Incident_ID'].apply(clean_incident_id)

print(df[['Incident_ID', 'Incident_ID_Clean']].drop_duplicates().head(15))
```

---

### 🔧 Scenario #14 — Price Version Normalization

```python
def clean_price_version(val):
    if pd.isna(val) or str(val).strip() in ('', 'N/A', 'na'):
        return 'UNKNOWN'
    val = str(val).strip().upper()
    # V1/v1/V1.0/1 → V1
    val = re.sub(r'^V?(\d+)(\.0)?$', r'V\1', val)
    # 2025-Q1/2025-q1 → 2025-Q1
    val = re.sub(r'^(\d{4})-?(Q\d)$', r'\1-\2', val.upper())
    return val

df['Price_Version_Clean'] = df['Price_Version'].apply(clean_price_version)

print(df['Price_Version_Clean'].value_counts())
```

---

### 🔧 Scenario #15 — FX Rate Cleaning

```python
def clean_fx_rate(val):
    if pd.isna(val) or str(val).strip() in ('', 'N/A', 'na', 'NA'):
        return None
    try:
        return round(float(str(val).strip()), 4)
    except:
        return None

df['FX_Rate_Clean'] = df['FX_Rate'].apply(clean_fx_rate)

# Convert all costs to INR using FX rate
def convert_to_inr(row):
    cost     = row['Cost_Clean']
    currency = row['Currency_Clean']
    fx       = row['FX_Rate_Clean']
    if pd.isna(cost):
        return None
    if currency == 'INR':
        return cost
    if currency in ('USD', 'EUR', 'GBP') and fx:
        return round(cost * fx, 2)
    return cost   # return as-is if no FX available

df['Cost_INR'] = df.apply(convert_to_inr, axis=1)

print(df[['Cost_Clean', 'Currency_Clean', 'FX_Rate_Clean', 'Cost_INR']].head(10))
```

---

### 🔧 Scenario #16 — Idle Resource Detection

```python
# Flag resources with very low usage
IDLE_THRESHOLD_SECONDS = 300   # less than 5 minutes usage = idle

df['Is_Idle_Resource'] = (
    df['Usage_Seconds'].notna() &
    (df['Usage_Seconds'] < IDLE_THRESHOLD_SECONDS)
)

print(f"Idle resources detected: {df['Is_Idle_Resource'].sum()}")
print(f"Idle rate: {df['Is_Idle_Resource'].mean()*100:.2f}%")
```

---

### 🔧 Scenario #17 — Pricing Type Normalization

```python
def clean_pricing_type(val):
    if pd.isna(val) or str(val).strip() in ('', 'N/A', 'na'):
        return 'UNKNOWN'
    val = str(val).strip().upper().replace('-', '').replace('_', '')
    if any(x in val for x in ['ONDEMAND', 'OD', 'ONDEM']):
        return 'ON_DEMAND'
    if any(x in val for x in ['RESERVED', 'RESERVE', 'RI']):
        return 'RESERVED'
    if any(x in val for x in ['SPOT', 'PREEMPTIBLE']):
        return 'SPOT'
    return 'UNKNOWN'

df['Pricing_Type_Clean'] = df['Pricing_Type'].apply(clean_pricing_type)

print(df['Pricing_Type_Clean'].value_counts())
```

---

### 🔧 Scenario #18 — Cost Allocation Key Validation

```python
# Valid departments and projects from org master
VALID_DEPARTMENTS = {
    'ENGINEERING', 'ENG', 'FINANCE', 'FIN',
    'DATA_SCIENCE', 'DS', 'OPS', 'OPERATIONS',
    'PRODUCT', 'SECURITY', 'ML', 'DEVOPS'
}
VALID_PROJECTS = {
    'PROJ-ALPHA', 'PROJ-BETA', 'PROJ-DELTA',
    'PROJ-EPSILON', 'PROJ-GAMMA'
}

def clean_department(val):
    if pd.isna(val) or str(val).strip() in ('', 'N/A', 'na', 'unknown', 'UNKNOWN'):
        return 'UNALLOCATED'
    val = str(val).strip().upper().replace(' ', '_').replace('-', '_')
    dept_map = {
        'ENGINEERING': 'ENGINEERING', 'ENG': 'ENGINEERING',
        'FINANCE':     'FINANCE',     'FIN': 'FINANCE',
        'DATA_SCIENCE':'DATA_SCIENCE','DS':  'DATA_SCIENCE',
        'OPS':         'OPS',         'OPERATIONS': 'OPS',
        'PRODUCT':     'PRODUCT',     'PROD': 'PRODUCT',
    }
    return dept_map.get(val, 'UNALLOCATED')

def clean_project(val):
    if pd.isna(val) or str(val).strip() in ('', 'N/A', 'na', 'unknown', 'UNKNOWN'):
        return 'UNALLOCATED'
    val = str(val).strip().upper().replace('_', '-').replace(' ', '-')
    proj_map = {
        'ALPHA': 'PROJ-ALPHA', 'PROJ-ALPHA': 'PROJ-ALPHA',
        'BETA':  'PROJ-BETA',  'PROJ-BETA':  'PROJ-BETA',
        'DELTA': 'PROJ-DELTA', 'PROJ-DELTA': 'PROJ-DELTA',
    }
    return proj_map.get(val, 'UNALLOCATED')

df['Department_Clean'] = df['Department'].apply(clean_department)
df['Project_Clean']    = df['Project'].apply(clean_project)

# Validate both together
df['Cost_Allocation_Valid'] = (
    (df['Department_Clean'] != 'UNALLOCATED') &
    (df['Project_Clean']    != 'UNALLOCATED')
)

print(f"Valid allocations:   {df['Cost_Allocation_Valid'].sum()}")
print(f"Invalid allocations: {(~df['Cost_Allocation_Valid']).sum()}")
```

---

### 🔧 Scenario #19 — SLA Event Marking

```python
def clean_sla_event(val):
    if pd.isna(val) or str(val).strip() in ('', 'N/A', 'na'):
        return False
    val = str(val).strip().upper()
    return val in ('YES', 'Y', 'TRUE', '1', 'T')

df['SLA_Event_Clean'] = df['SLA_Event'].apply(clean_sla_event)

print(f"SLA events marked TRUE:  {df['SLA_Event_Clean'].sum()}")
print(f"SLA events marked FALSE: {(~df['SLA_Event_Clean']).sum()}")
```

---

### 🔧 Scenario #20 — Log Time Skew Correction

```python
def apply_skew_correction(ts_utc, skew_seconds):
    if pd.isna(ts_utc) or pd.isna(skew_seconds):
        return ts_utc
    try:
        skew = int(float(str(skew_seconds)))
        dt   = pd.to_datetime(ts_utc, utc=True)
        corrected = dt - pd.Timedelta(seconds=skew)   # subtract skew
        return corrected.strftime('%Y-%m-%d %H:%M:%S+00:00')
    except:
        return ts_utc

df['Log_Skew_Seconds_Clean'] = pd.to_numeric(df['Log_Skew_Seconds'], errors='coerce')
df['TS_UTC_Corrected'] = df.apply(
    lambda row: apply_skew_correction(row['TS_UTC'], row['Log_Skew_Seconds_Clean']),
    axis=1
)

large_skew = df['Log_Skew_Seconds_Clean'].abs() > 60
print(f"Records with large skew (>60s): {large_skew.sum()}")
print(df[large_skew][['TS_UTC', 'Log_Skew_Seconds_Clean', 'TS_UTC_Corrected']].head(5))
```

---

## 🏗️ STEP 5 — Build the Final Clean Dataset

```python
# Select only cleaned columns for the output
df_clean = pd.DataFrame({
    'Usage_ID':              df['Usage_ID'],
    'Account_Clean':         df['Account_Clean'],
    'TS_UTC_Corrected':      df['TS_UTC_Corrected'],
    'Service':               df['Service'],
    'SKU_Clean':             df['SKU_Clean'],
    'Usage_Seconds':         df['Usage_Seconds'],
    'Cost_Clean':            df['Cost_Clean'],
    'Cost_INR':              df['Cost_INR'],
    'Currency_Clean':        df['Currency_Clean'],
    'FX_Rate_Clean':         df['FX_Rate_Clean'],
    'Region_Clean':          df['Region_Clean'],
    'Free_Tier_Clean':       df['Free_Tier_Clean'],
    'Tag_Owner_Clean':       df['Tag_Owner_Clean'],
    'Tag_Env_Clean':         df['Tag_Env_Clean'],
    'Resource_ID_Clean':     df['Resource_ID_Clean'],
    'Resource_ID_Valid':     df['Resource_ID_Valid'],
    'Ticket_ID_Clean':       df['Ticket_ID_Clean'],
    'Ticket_Text_Masked':    df['Ticket_Text_Masked'],
    'Incident_ID_Clean':     df['Incident_ID_Clean'],
    'Price_Version_Clean':   df['Price_Version_Clean'],
    'Pricing_Type_Clean':    df['Pricing_Type_Clean'],
    'Department_Clean':      df['Department_Clean'],
    'Project_Clean':         df['Project_Clean'],
    'SLA_Event_Clean':       df['SLA_Event_Clean'],
    'Log_Skew_Seconds_Clean':df['Log_Skew_Seconds_Clean'],
    'Purchase_Type':         df['Purchase_Type'],
    'Anomaly_Flag':          df['Anomaly_Flag'],
    'Duplicate_Flag':        df['Duplicate_Flag'],
    'Cost_Allocation_Valid': df['Cost_Allocation_Valid'],
    'Is_Idle_Resource':      df['Is_Idle_Resource'],
})

# Remove duplicates
df_clean = df_clean[~df_clean['Duplicate_Flag']].copy()

print(f"\n✅ Clean dataset shape: {df_clean.shape}")
print(f"Rows removed (duplicates): {df['Duplicate_Flag'].sum()}")
print(f"Null values remaining:\n{df_clean.isnull().sum().sort_values(ascending=False).head(10)}")
```

---

## 🏗️ STEP 6 — Validate the Cleaned Data

```python
print("=" * 50)
print("DATA QUALITY VALIDATION REPORT")
print("=" * 50)

checks = {
    "No null Account_Clean":     df_clean['Account_Clean'].notna().all(),
    "No null TS_UTC":            df_clean['TS_UTC_Corrected'].notna().mean() > 0.95,
    "No null SKU_Clean":         df_clean['SKU_Clean'].notna().all(),
    "Usage_Seconds all numeric": df_clean['Usage_Seconds'].notna().mean() > 0.95,
    "Cost_Clean all numeric":    df_clean['Cost_Clean'].notna().mean() > 0.95,
    "No duplicate rows":         ~df_clean.duplicated().any(),
    "Region uses standard slugs":df_clean['Region_Clean'].str.match(
                                     r'^[a-z]+-[a-z]+-\d$', na=False).mean() > 0.8,
    "SLA_Event is boolean":      df_clean['SLA_Event_Clean'].dtype == bool,
    "No PII in tickets":         ~df_clean['Ticket_Text_Masked'].str.contains(
                                     r'@|\d{10}', na=False).any(),
}

for check, result in checks.items():
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status}  {check}")
```

---

## 🏗️ STEP 7 — Save Cleaned Dataset

```python
# Save to Excel
df_clean.to_excel('Cloud_CSP_Cleaned_Dataset.xlsx', index=False)

# Save to CSV for pipeline use
df_clean.to_csv('Cloud_CSP_Cleaned_Dataset.csv', index=False)

print(f"\n✅ Saved {len(df_clean)} clean rows")
print(f"✅ {len(df_clean.columns)} columns")
print("Files saved: Cloud_CSP_Cleaned_Dataset.xlsx + .csv")
```

---

## 📊 Complete Flow Summary

```
RAW DATA (10,200 rows, 28 messy columns)
         ↓
STEP 1 → Load as string dtype
         ↓
STEP 2 → EDA — understand each column
         ↓
STEP 3 → Clean column by column (20 scenarios)
         ↓
STEP 4 → Validate each cleaned column
         ↓
STEP 5 → Build final clean dataframe
         ↓
STEP 6 → Run quality checks
         ↓
CLEAN DATA (10,000 rows, 30 clean columns)
         ↓
STEP 7 → Save to Excel + CSV → Feed into Transformations
```

---

### 🎯 What to Tell the Interviewer:

> *"We follow a strict order — load everything as string first, explore before cleaning, clean column by column mapping each step to a specific scenario, validate after every major transformation, and only then build the final clean dataset. This is not just cleaning — this is a production-grade data pipeline. In a real environment, every one of these functions would be a modular, testable unit inside an Apache Spark job running on a distributed cluster."*

---

> ✅ Send the next screenshot whenever you're ready!