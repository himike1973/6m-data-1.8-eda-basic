# 📚 Lesson 1.8: EDA Basic — Exploratory Data Analysis

## Session Overview

| | |
|---|---|
| **Duration** | 3 hours |
| **Format** | Flipped Classroom + Guided Coding in Jupyter |
| **Tools** | VS Code + `pds` conda environment |
| **Notebook** | `notebooks/eda_basic.ipynb` |

## Agenda

| Time | Part | Topic |
|------|------|-------|
| 0:00 – 1:00 | Part 1 | Descriptive Statistics — `.info()`, `.describe()`, `.value_counts()` |
| 1:00 – 2:00 | Part 2 | Data Quality — missing values, duplicates, outliers |
| 2:00 – 3:00 | Part 3 | Data Transformation — string & categorical ops, file I/O, regex |

## 🎯 Learning Objectives

By the end of this session, you will be able to:

1. Summarise a dataset using descriptive statistics and identify its shape, data types, and distributions.
2. Handle missing values, duplicates, and outliers using appropriate Pandas methods.
3. Transform data through type conversion, string cleaning, and categorical encoding.
4. Read and write data across multiple file formats (CSV, JSON, Excel, databases).

---

## Before You Start

**Have you completed the pre-class reading?**
- ✓ Understand what EDA is and why it matters
- ✓ Review basic statistics concepts (mean, median, standard deviation)
- ✓ Familiar with regex basics
- ✓ `pds` conda environment is set up

Open the notebook in VSCode by double-clicking on `notebooks/eda_basic.ipynb`, then select the `pds` conda environment for the kernel.

---

## 🏃 Part 1: Exploratory Data Analysis — Basic

Open the notebook and follow along with **Part 1**.

This part covers the "Health Check" of a dataset — understanding what you have before you analyse it.

**Key methods you will use:**
- `.info()` — data types, null counts, memory usage
- `.describe()` — summary statistics for numeric columns
- `.value_counts()` — frequency counts for categorical columns
- `.shape`, `.columns`, `.dtypes` — structural overview

> Work through each notebook cell, run the code, and read the output carefully. The goal is to be able to answer: "What is in this dataset, and is it ready for analysis?"

---

## 🏃 Part 2: Data Quality

Continue in the notebook with **Part 2**.

**Key topics:**
- Detecting and handling missing values (`isna()`, `fillna()`, `dropna()`)
- Finding and removing duplicates (`duplicated()`, `drop_duplicates()`)
- Identifying outliers (IQR method, z-score)
- Cleaning strategies — when to fill, when to drop

> Pay attention to the trade-offs: filling missing values preserves rows but introduces assumptions; dropping rows preserves accuracy but reduces sample size.

---

## 🏃 Part 3: Data Transformation

Continue in the notebook with **Part 3**.

**Key topics:**
- Type conversion (`astype()`, `pd.to_datetime()`, `pd.to_numeric()`)
- String operations (`.str.lower()`, `.str.strip()`, `.str.replace()`)
- Categorical encoding (`pd.Categorical()`, `pd.get_dummies()`)
- Regex basics for pattern matching
- Reading and writing files: CSV, JSON, Excel, SQLite

---

## 🎯 Wrap-Up

**Key Takeaways:**
1. Always run a "Health Check" on a new dataset before any analysis — shape, types, nulls, distributions.
2. Data cleaning decisions (fill vs. drop) depend on the context and your analysis goals.
3. Type conversion and string cleaning are foundational — messy types cause silent errors downstream.

**Next Steps:**
- Complete the [Assignment](./assignment.md) — EDA practice challenges: clean and analyse a messy dataset.
- Next lesson: Lesson 1.9 covers EDA Advanced — time series, merging datasets, GroupBy, and pivot tables.
