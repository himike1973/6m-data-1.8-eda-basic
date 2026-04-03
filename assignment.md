# 📝 Post-Class: EDA Basic

> ⏱️ **Estimated Time:** 45–60 minutes | Complete this **after** your class session.

---

## 🎯 Learning Objectives Revisited

This assignment reinforces what you practised in class:
- Summarising datasets with descriptive statistics
- Handling missing values, duplicates, and outliers
- Transforming string and categorical data

---

## Part 1: Conceptual Check (15 min)

**Question 1:** How do you select rows from a DataFrame where **any value in the row** exceeds a given threshold?

**Question 2:** How do you sort a DataFrame by column `A` ascending and column `B` descending simultaneously?

**Question 3:** How do you concatenate two DataFrames vertically and reset the index?

**Question 4:** What is the difference between removing outliers and capping them? When would you choose each approach?

**Question 5:** You call `df.fillna(df.mean())` on a DataFrame that has both numeric and categorical columns. What happens, and is there a better approach?

<details>
<summary>💡 Check Your Answers</summary>

**Q1:** `df[df.max(axis=1) > threshold]` — computes the row-wise maximum and filters rows where it exceeds the threshold.

**Q2:** `df.sort_values(['A', 'B'], ascending=[True, False])`

**Q3:** `pd.concat([df1, df2], ignore_index=True)` — `ignore_index=True` resets the index from 0.

**Q4:** **Removing** permanently deletes the outlier rows, which can reduce your dataset size significantly. **Capping** (clipping to a max/min threshold) preserves the row but limits the extreme value. Remove when outliers are clear data errors; cap when they're legitimate but distort analysis.

**Q5:** `.fillna(df.mean())` will only work on numeric columns — categorical columns will be ignored and remain NaN. A better approach: `df.fillna({'numeric_col': df['numeric_col'].mean(), 'cat_col': df['cat_col'].mode()[0]})`

</details>

---

## Part 2: Practical Challenge (30–45 min)

### Scenario: "The Patient Records Audit"

A hospital has given you a messy export of patient records. Your job is to clean it and produce a summary report before it can be used for any analysis.

---

### Challenge 1: The Health Check

```python
import pandas as pd
import numpy as np

# Simulated messy hospital data
data = {
    'patient_id': [101, 102, 103, 102, 104, 105, 106],
    'name': ['Alice', 'Bob', 'Charlie', 'Bob', 'Diana', None, 'Frank'],
    'age': [34, 28, 150, 28, 45, 62, 29],  # Note: 150 is impossible
    'blood_type': ['A+', 'O-', 'B+', 'O-', None, 'AB+', 'A-'],
    'ward': ['Cardiology', 'neurology', 'Cardiology', 'neurology', 'Oncology', 'Cardiology', 'neurology'],
    'admission_date': ['2024-01-15', '2024-01-20', '2024-02-01', '2024-01-20', '2024-02-10', '2024-02-15', '2024-03-01'],
    'days_admitted': [5, 3, 8, 3, None, 12, 4]
}

df = pd.DataFrame(data)
```

**Tasks:**
1. Run `.info()` — which columns have missing values?
2. Run `.describe()` — identify the outlier in the `age` column.
3. Use `.isnull().sum()` to count missing values per column.
4. Use `.duplicated().sum()` to count duplicate rows.

<details>
<summary>💡 Hint</summary>

A duplicated row is one where **all** column values are identical. You can also check specific columns: `df.duplicated(subset=['patient_id']).sum()`

</details>

---

### Challenge 2: Clean the Data

**Tasks:**
1. Remove the duplicate row (keep the first occurrence).
2. Fix the impossible age (150) — replace it with the column median.
3. Standardise the `ward` column — all values should be title case (`'Cardiology'`, not `'cardiology'`).
4. Fill the missing `blood_type` with the string `'Unknown'`.
5. Fill missing `days_admitted` with the column median.

<details>
<summary>✅ Solution</summary>

```python
# 1. Remove duplicates
df = df.drop_duplicates()

# 2. Fix impossible age
median_age = df.loc[df['age'] <= 120, 'age'].median()
df.loc[df['age'] > 120, 'age'] = median_age

# 3. Standardise ward names
df['ward'] = df['ward'].str.title()

# 4. Fill missing blood type
df['blood_type'] = df['blood_type'].fillna('Unknown')

# 5. Fill missing days admitted
df['days_admitted'] = df['days_admitted'].fillna(df['days_admitted'].median())

print(df)
```

</details>

---

### Challenge 3: Transform & Summarise

**Tasks:**
1. Convert `admission_date` to datetime type.
2. Add a `risk_level` column using `.apply()`: age > 60 → `'High'`; age 40–60 → `'Medium'`; under 40 → `'Low'`.
3. Use `.value_counts()` to count patients per `ward`.
4. Compute the average `days_admitted` per `ward`.

<details>
<summary>✅ Solution</summary>

```python
# 1. Convert date
df['admission_date'] = pd.to_datetime(df['admission_date'])

# 2. Risk level
def risk_level(age):
    if age > 60:
        return 'High'
    elif age >= 40:
        return 'Medium'
    else:
        return 'Low'

df['risk_level'] = df['age'].apply(risk_level)

# 3. Patients per ward
print(df['ward'].value_counts())

# 4. Average days per ward
print(df.groupby('ward')['days_admitted'].mean().round(1))
```

</details>

---

### 🏆 Stretch Challenge (Optional)

Export the cleaned DataFrame to a CSV file, then reload it and verify:
- All data types are preserved (especially the datetime column).
- Row count matches what you expect.

> **Hint:** When reloading, you'll need to parse the date column again with `parse_dates=['admission_date']`.

---

## 💬 Reflection (5 min)

In 2–3 sentences, answer:

> *Think about the "150 years old" data point — was it right to replace it with the median? What information was lost by doing so? In a real hospital, what would the responsible action be?*

---

## 📤 Share Your Work

Post your cleaned DataFrame's `.info()` output and your reflection in **Discord** under `#assignments`.
