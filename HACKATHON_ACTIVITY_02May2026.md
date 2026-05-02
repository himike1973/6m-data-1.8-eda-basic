# NumPy / Pandas / EDA Hackathon — Singapore Talent Market Intelligence

**Duration:** 2 hours of analysis, followed by team presentations **Tooling:** Python (NumPy, Pandas, Matplotlib/Seaborn) **Dataset:** `SGJobData.csv` (\~1.05M Singapore job postings, Oct 2022 – May 2024\) **Format:** Teams of 5–6 preferably

---

## 1\. Why this activity exists

Last week you interrogated this dataset using SQL. This week you are doing it again — but with a completely different toolset. NumPy, Pandas, and EDA are how working data analysts actually spend their days: loading messy CSVs, shaping data into tidy DataFrames, computing summaries with arrays, and turning numbers into charts that tell a story.

By the end of this session you will have practised:

- **NumPy fundamentals:** vectorised arithmetic, boolean masking, statistical functions (`mean`, `median`, `std`, `percentile`), and working with numerical arrays extracted from a DataFrame.  
- **Pandas fundamentals:** loading and inspecting a large CSV, selecting and filtering rows/columns, groupby aggregations, handling missing values, string methods, and merging/reshaping data.  
- **EDA workflow:** profiling a new dataset systematically — shape, dtypes, nulls, distributions, outliers, correlations — before asking any business question.  
- **Business communication:** turning a notebook of outputs into a 5-minute story a non-technical stakeholder can act on.

**The hardest part is not the code. It is knowing which question to ask, and which chart actually answers it.** Spend the first 10 minutes of your team time arguing about the question, not writing code.

---

## 2\. Schedule (2 hours, then presentations)

| Time | Phase | What you are doing |
| :---- | :---- | :---- |
| 0:00 – 0:15 | **Kick-off & setup** | Form teams, load the data into a Jupyter notebook, run the profile block in §4, pick a persona. |
| 0:15 – 0:35 | **Warm-up: Profile the data** | Everyone answers the 6 warm-up questions in §7. This builds shared intuition. |
| 0:35 – 1:35 | **Deep dive: Persona challenge** | Work on your persona brief in §8. Define questions, write the code, capture findings. |
| 1:35 – 2:00 | **Build the story** | Decide your top 3 insights \+ 1 surprise. Prepare a 5-minute pitch. A clean Jupyter notebook with output cells visible is enough. |
| 2:00 – end | **Presentations** | 5 minutes per team \+ 2 minutes Q\&A. |

---

## 3\. Setup (do this in the first 5 minutes)

Open a Jupyter notebook. The file is large (\~286 MB), so load it once and keep the DataFrame in memory.

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load — this may take 10–15 seconds on the first run
df = pd.read_csv('SGJobData.csv', low_memory=False)

# Sanity checks
print(df.shape)          # Expect (~1048585, 20)
print(df.dtypes)
print(df.head(3))
```

**Date columns need explicit parsing** — do this early so date arithmetic works later:

```py
df['metadata_originalPostingDate'] = pd.to_datetime(df['metadata_originalPostingDate'])
df['metadata_newPostingDate']       = pd.to_datetime(df['metadata_newPostingDate'])
df['metadata_expiryDate']           = pd.to_datetime(df['metadata_expiryDate'])
```

---

## 4\. Data dictionary

| Column | Type | Notes |
| :---- | :---- | :---- |
| `categories` | str (JSON array) | e.g. `[{"id":21,"category":"Information Technology"}]` — a job can belong to **multiple** categories. You will need to parse this. |
| `employmentTypes` | str | Permanent, Full Time, Contract, Part Time, Temporary, Internship/Attachment, Freelance, Flexi-work. |
| `metadata_expiryDate` | date | When the posting expires. |
| `metadata_isPostedOnBehalf` | bool | True if a recruiter posted on behalf of the hiring company. |
| `metadata_jobPostId` | str | Unique ID, e.g. `MCF-2023-0252866`. |
| `metadata_newPostingDate` | date | Date of the most recent re-post. |
| `metadata_originalPostingDate` | date | Date the job was first posted. |
| `metadata_repostCount` | int | How many times the same job has been re-posted. A signal of hard-to-fill roles. |
| `metadata_totalNumberJobApplication` | int | Number of applications received. |
| `metadata_totalNumberOfView` | int | Number of times the post was viewed. |
| `minimumYearsExperience` | int | Years of experience required. |
| `numberOfVacancies` | int | Open headcount. |
| `positionLevels` | str | Fresh/entry, Junior Executive, Executive, Senior Executive, Professional, Manager, Middle Management, Senior Management, Non-executive. |
| `postedCompany_name` | str | The poster (often a recruitment agency, **not** the hiring employer). |
| `salary_minimum` / `salary_maximum` | int | Salary band. |
| `salary_type` | str | Almost all `Monthly`. |
| `status_jobStatus` | str | Open, Closed, Re-open. |
| `title` | str | Free-text job title. |
| `average_salary` | float | Pre-computed mean of min/max. |

---

## 5\. Known data-quality landmines (read this\!)

Real datasets are dirty. Decide your handling and **state it in your presentation**.

1. **Salary outliers.** `salary_maximum` goes up to **$25,330,000/month**. These are data-entry errors. Decide your cap (e.g. exclude rows where `average_salary > 50_000`, or clip at the 99th percentile using `np.percentile`).  
2. **Recruitment agencies dominate.** The top "companies" are agencies, not employers. If your question is "who is hiring?", agency-posted jobs distort the answer. Use `metadata_isPostedOnBehalf` and string filtering on `postedCompany_name`.  
3. **`categories` is a JSON string, not a list.** A job can have 1–N categories. You will need `str.extract`, `str.findall`, or `ast.literal_eval` to parse it.  
4. **Re-posts inflate row counts.** A single role re-posted 5 times shows up as 5 rows. Use `drop_duplicates(subset='metadata_jobPostId', keep='first')` when counting unique roles.  
5. **`salary_minimum = 1`** appears frequently — that is a placeholder, not a real salary. Filter it out with `df[df['salary_minimum'] > 100]`.  
6. **Mixed nulls.** Columns like `positionLevels`, `employmentTypes`, and `salary_type` contain `None` as a string, actual `NaN`, and legitimate blanks. Use `.replace('None', np.nan)` before `.isnull()` checks.  
7. **Date coverage is uneven** at the start and end of the range. Trim to whole months when computing trends.

---

## 6\. Python cheat sheet for this dataset

Most teams will get stuck on the same things. Bookmark this section.

**Loading and initial profiling:**

```py
# Shape, types, missing values at a glance
print(f"Rows: {df.shape[0]:,}  Columns: {df.shape[1]}")
print("\nNull counts:\n", df.isnull().sum().sort_values(ascending=False).head(10))
print("\nBasic stats:\n", df[['salary_minimum','salary_maximum','average_salary',
                               'minimumYearsExperience','numberOfVacancies']].describe())
```

**Salary cleaning with NumPy:**

```py
# Remove obvious outliers using the 99th percentile
p99 = np.percentile(df['average_salary'].dropna(), 99)
df_clean = df[(df['average_salary'] > 500) & (df['average_salary'] <= p99)].copy()

# Use NumPy to compute stats on the cleaned array
sal = df_clean['average_salary'].to_numpy()
print(f"Mean: {np.mean(sal):,.0f}  Median: {np.median(sal):,.0f}  Std: {np.std(sal):,.0f}")
print(f"25th pct: {np.percentile(sal, 25):,.0f}  75th pct: {np.percentile(sal, 75):,.0f}")
```

**Groupby aggregation — salary by position level:**

```py
salary_by_level = (
    df_clean
    .groupby('positionLevels')['average_salary']
    .agg(['mean', 'median', 'count'])
    .rename(columns={'mean': 'avg_salary', 'median': 'median_salary', 'count': 'num_jobs'})
    .sort_values('median_salary', ascending=False)
)
print(salary_by_level)
```

**Parsing the `categories` JSON string (one row → first category):**

```py
import re

def extract_first_category(cat_str):
    """Extract the first category label from the JSON-like string."""
    if pd.isna(cat_str):
        return np.nan
    match = re.search(r'"category"\s*:\s*"([^"]+)"', str(cat_str))
    return match.group(1) if match else np.nan

df['primary_category'] = df['categories'].apply(extract_first_category)
print(df['primary_category'].value_counts().head(10))
```

**Monthly posting trend:**

```py
df_clean['year_month'] = df_clean['metadata_originalPostingDate'].dt.to_period('M')
monthly = df_clean.groupby('year_month').size().reset_index(name='postings')

plt.figure(figsize=(12, 4))
plt.plot(monthly['year_month'].astype(str), monthly['postings'])
plt.xticks(rotation=45, ha='right')
plt.title('Monthly Job Postings — SGJobData')
plt.tight_layout()
plt.show()
```

**Filtering out recruitment agencies (rough heuristic):**

```py
agency_keywords = ['RECRUIT', 'HR ADVISORY', 'MANPOWER', 'STAFFING', 'CONSULT', 'TALENT']
pattern = '|'.join(agency_keywords)

direct_only = df_clean[
    (~df_clean['postedCompany_name'].str.upper().str.contains(pattern, na=False)) &
    (df_clean['metadata_isPostedOnBehalf'] == False)
]
```

**Correlation heatmap:**

```py
num_cols = ['average_salary', 'minimumYearsExperience', 'numberOfVacancies',
            'metadata_repostCount', 'metadata_totalNumberJobApplication',
            'metadata_totalNumberOfView']
corr = df_clean[num_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix — Numerical Columns')
plt.tight_layout()
plt.show()
```

**Boxplot for outlier inspection:**

```py
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_clean, x='positionLevels', y='average_salary')
plt.xticks(rotation=45, ha='right')
plt.title('Salary Distribution by Position Level')
plt.tight_layout()
plt.show()
```

---

## 7\. Warm-up (everyone, 20 minutes)

Answer these six together as a team. One code block each. Capture your outputs as visible notebook cells — you may refer to them in your presentation.

1. What is the **shape** of the DataFrame, the **date range** of `metadata_originalPostingDate`, and how many **unique job post IDs** are there?  
2. What are the **top 10 most common `primary_category` values**, and what percentage of all jobs does each represent? *(Use `.value_counts(normalize=True)`)*  
3. After removing salary outliers, what is the **median `average_salary`** broken down by `positionLevels`? Sort from highest to lowest.  
4. Which **5 columns have the most missing values**? What percentage of rows are null in each?  
5. What is the **distribution of `minimumYearsExperience`**? Plot a histogram. Where does the bulk of demand sit?  
6. Compute the **correlation** between `average_salary`, `minimumYearsExperience`, `metadata_repostCount`, and `metadata_totalNumberJobApplication`. Which pair has the strongest relationship?

---

## 8\. Persona challenges (pick ONE per team)

Each team adopts a stakeholder. The brief is **deliberately vague** — your first job is to translate it into 3–5 specific, answerable questions, then write the code. **You will be assessed as much on the questions you frame as on the Pandas you write.**

### Persona A — The Career Coach

*"I advise mid-career professionals who want to switch into tech, healthcare, or finance. I need evidence-based advice: which switches are realistic, what salary they should target, and what experience employers actually demand."*

- Compare salary distributions (box plots or violin plots) across your three target categories.  
- Identify "low barrier, high demand" roles: categories with many postings AND low median `minimumYearsExperience`. Use a scatter plot — one dot per category.  
- Surface the 10 fastest-growing job titles over the dataset's time range. Use monthly `groupby` \+ a rolling window or percentage change.

### Persona B — The Salary Benchmarking Analyst

*"My company sells salary benchmarking reports to HR teams. Build me a defensible view of pay by role, level, and experience that I can actually publish."*

- Robust outlier handling is critical — defend your cleaning rules in the notebook (a Markdown cell is fine).  
- Produce a pivot table: `positionLevels` × `primary_category` showing **median** salary. Which cell is the highest-paid intersection?  
- Use NumPy to compute the **inter-quartile range (IQR)** for each position level and flag levels where the IQR is unusually wide. What does a wide IQR signal?  
- Plot a line or bar chart showing how median salary changes with `minimumYearsExperience` (0–10 years only).

### Persona C — The Government Workforce Planner

*"The Ministry of Manpower wants to know which sectors face structural hiring difficulty so we can target retraining grants."*

- Create a "difficulty score" using NumPy: combine `metadata_repostCount`, `numberOfVacancies`, and `metadata_totalNumberJobApplication` into a single normalised index (hint: `(x - x.min()) / (x.max() - x.min())`).  
- Which categories and position levels have the highest difficulty scores?  
- Are there roles with high vacancy counts but **very low application rates** (applications ÷ views)? Visualise with a scatter plot.  
- Plot monthly `metadata_repostCount` trends by category for the top 5 hard-to-fill sectors.

### Persona D — The Job Seeker Optimisation Coach

*"I help job seekers compete. I want to know: when to apply, what title patterns get the most views, and where applications are most likely to convert."*

- Use `dt.day_name()` and `dt.month` on `metadata_originalPostingDate` to find which posting days and months attract the most views and applications.  
- Use `str.contains` to flag titles with keywords like `Senior`, `Junior`, `Lead`, `Specialist`, `Manager`. Compare median views and application counts across these groups.  
- Which `primary_category` has the lowest applications-per-vacancy ratio (least competitive)? Visualise as a horizontal bar chart.

### Persona E — The Recruitment Agency Owner

*"I run a mid-sized agency. I want to know which segments my competitors dominate, and where there is white space we could move into."*

- Profile the top 20 agencies (filter `metadata_isPostedOnBehalf == True`): what categories, position levels, and salary bands do they specialise in? Use a heatmap of job counts.  
- Compare agency-posted vs direct-posted jobs on median views and median application counts using a grouped bar chart.  
- Find under-served segments: categories where direct-employer posting share is high (agencies barely present). Are salaries different there?

### Persona F — The Tech Sector Strategist

*"I lead strategy at a tech company. I need a state-of-the-Singapore-tech-job-market report: who's hiring, for what skills, at what level, and how that has shifted over the last 18 months."*

- Filter to `primary_category == 'Information Technology'` (or similar). How many unique postings? What share of the whole market?  
- Use `str.contains` on the `title` column to tag roles: `Engineer`, `Data`, `DevOps`, `Cloud`, `Cybersecurity`, `AI/ML`. Plot a stacked bar chart of monthly demand by tag.  
- Show median salary trajectory over time (monthly) for the top 4 tech sub-roles. Has post-2023 demand shifted?  
- Which position levels are growing in tech vs declining? Use `groupby(['year_month', 'positionLevels']).size()` and a line chart.

---

## 9\. Technical requirements (your notebook must demonstrate ALL of these)

To prove you stretched your Python muscles, your final notebook must include:

- [ ] At least **one NumPy operation** applied to a column extracted as an array (e.g. `np.mean`, `np.percentile`, `np.std`, boolean mask, or a normalisation formula).  
- [ ] At least **one Pandas groupby** with a meaningful aggregation (not just `.count()`) — e.g. `.agg({'col': ['mean', 'median']})`.  
- [ ] At least **one missing-value handling step** — `isnull()` inspection plus a `.fillna()`, `.dropna()`, or `.replace()` decision that is **documented in a Markdown cell**.  
- [ ] At least **one string/text operation** on the `title` or `positionLevels` or `categories` column (`.str.contains`, `.str.extract`, `.str.upper`, etc.).  
- [ ] At least **one date/time operation** using the `metadata_originalPostingDate` column (`dt.month`, `dt.year`, `dt.to_period`, etc.).  
- [ ] At least **two different chart types** (e.g. bar \+ boxplot, histogram \+ scatter, line \+ heatmap) — each with a title, axis labels, and a one-sentence interpretation comment below.  
- [ ] A **documented data-cleaning step** in a Markdown cell: what you removed, why, and how many rows were affected.

Be ready to point to the cell in your notebook where each of the above appears.

---

## 10\. Deliverables (what you present)

A 5-minute pitch covering:

1. **The question** you chose to answer (one sentence) and **why it matters** to your persona.  
2. **3 headline insights**, each backed by a number and the chart or code that produced it.  
3. **1 surprising finding** — something you did not expect when you started.  
4. **Data caveats** — what you cleaned, excluded, or could not answer with this data alone.  
5. **Recommendation** — one concrete action your persona should take next week.

You do **not** need a slide deck. A Jupyter notebook with clean output cells and a one-page bullet sheet is enough. **Walk us through the notebook** during your pitch.

---

## 11\. Grading rubric (out of 25\)

| Dimension | Weight | What we look for |
| :---- | :---- | :---- |
| **Question framing** | 5 | Was the question sharp, persona-relevant, and answerable from the data? |
| **Python craft** | 5 | Clean, readable code. Appropriate use of NumPy arrays, Pandas methods, and EDA patterns. No unnecessary loops where vectorised operations exist. |
| **Data-quality handling** | 5 | Outliers, nulls, agency rows, date parsing — handled and **defended** in Markdown. |
| **Insight quality** | 5 | Are the findings non-obvious, quantified, and actionable? Do the charts actually support the claims? |
| **Communication** | 5 | Could a non-technical stakeholder understand and act on this in 5 minutes? |

**Bonus point** for the team that finds the most genuinely *surprising* result — voted by the rest of the cohort.

---

## 12\. Stretch challenges (if you finish early)

- **Salary band width analysis:** For each `primary_category`, compute `salary_maximum - salary_minimum` using NumPy. Which categories have the widest bands? What might that signal about negotiation room?  
- **"Ghost jobs" revisited:** Using Pandas date arithmetic (`metadata_expiryDate - metadata_originalPostingDate`), identify postings open for more than 90 days with zero applications. How prevalent are they? Which categories and levels?  
- **Title keyword co-occurrence:** Build a frequency table of two-word title bigrams (split `title` on spaces, then use `pd.Series(bigrams).value_counts()`). Which bigrams are trending upward over time?  
- **Cohort analysis:** Group jobs by posting quarter. For each cohort, track median salary, median `minimumYearsExperience`, and application rate. Has the market become more or less demanding over time?  
- **Agency concentration index:** For each `primary_category`, compute the share of jobs posted by the top 3 agencies. Build a "concentration index" and rank categories from most agency-dominated to least.

---

## 13\. Comparing with last week

You analysed this same dataset with SQL last week. Use that experience to your advantage — and reflect on it.

- Which analytical questions were **easier** to express in SQL? Which are **easier** in Pandas?  
- Your SQL results are ground truth: if your Pandas output gives a different number for the same question, **debug before you present**.  
- SQL excels at set-based aggregations and joins; Pandas excels at row-by-row transformations, string manipulation, and plotting. Use the right tool for the right job — even if this week's rule is "Python only".

---

## 14\. Ground rules

- **Python only.** NumPy, Pandas, Matplotlib, Seaborn, and the Python standard library. No SQL, no Excel, no manual edits to the CSV.  
- **Work as a team.** Pair on hard cells. Split the persona's sub-questions across team members, then merge your notebooks at the end.  
- **Cite your numbers.** Every claim in your pitch must trace back to a visible output cell.  
- **Time-box ruthlessly.** Better to land 3 solid insights than 7 half-baked ones.  
- **Ask for help.** Instructors are here to unblock you on syntax — but the analytical choices are yours.

Good luck. You already know what is in this dataset. Now show us what Python can do with it.  
