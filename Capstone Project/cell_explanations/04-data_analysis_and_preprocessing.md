# 04 Data analysis and preprocessing

Overview

This cell reads a sample of the Amazon apparel reviews CSV, cleans text fields, engineers features, performs exploratory data analysis, and creates visualizations. It produces a data frame with several engineered features ready for modeling.

Purpose

- Load raw data used throughout the project.
- Clean and normalize textual fields.
- Create time and numeric features for analysis.
- Produce plots and simple statistics for presentation.

Line by line explanation

1. `import pandas as pd`, `import numpy as np`
   - Libraries for tabular data and numerical operations.

2. `import re` and `from bs4 import BeautifulSoup`
   - `re` is used for pattern based string cleaning. `BeautifulSoup` is used to strip HTML tags from text if present.

3. `import seaborn as sns`, `import matplotlib.pyplot as plt` and `from scipy import stats` and `from sklearn.preprocessing import MinMaxScaler, OneHotEncoder`
   - Visualization and statistical tools, plus scalers and encoders for feature preparation.

4. `df = pd.read_csv("amazon_reviews_us_Apparel_v1_00.csv", nrows=50000)`
   - Read the first 50,000 rows from the CSV for faster iteration during analysis.

5. `df = df.dropna(subset=["review_body", "star_rating", "verified_purchase"])`
   - Remove rows missing core columns required by the analysis.

6. `df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce', dayfirst=True)`
   - Convert the review date to a datetime. Invalid dates become NaT.

7. `df['review_year'] = df['review_date'].dt.year` and similar lines
   - Extract year, month, and day of week for time based exploratory analysis.

8. `def clean_text(text):` and the body
   - Remove HTML, non-alphanumeric characters except common punctuation, collapse whitespace, and lowercase the text. This produces consistent cleaned text for token counts and modeling.

9. `df['cleaned_review_body'] = df['review_body'].apply(clean_text)`
   - Apply the cleaning function to each review.

10. Summary stats and prints
   - `print(df[['star_rating', 'helpful_votes', 'total_votes']].describe())` prints central tendency and spread for the numeric columns.

11. `df['verified_purchase_binary'] = df['verified_purchase'].map({'Y': 1, 'N': 0})`
   - Convert the verified purchase flag into a numeric column for modeling.

12. Correlation matrix
   - Compute and print correlations among `star_rating`, `helpful_votes`, `total_votes`, and `verified_purchase_binary` to help assess relationships.

13. Most and least reviewed products
   - Use `value_counts()` on `product_id` to find which products have many or few reviews and print the top and bottom lists.

14. Plotting sections
   - Several plots are produced: rating distribution boxplots, bar chart of top categories, heatmap of correlations, time series of review volume, and pair plots of key features. Each plot uses seaborn and matplotlib with clear titles and labels.

15. `df['review_length'] = df['cleaned_review_body'].apply(lambda x: len(str(x).split()))`
   - Count words in the cleaned review and store it as a numeric feature.

16. Feature conversions and scaling
   - Convert boolean flags into numeric columns and one hot encode `product_category` using `pd.get_dummies`. Then apply `MinMaxScaler` to normalize `total_votes`, `helpful_votes`, and `review_length`.

17. Print engineered feature sample
   - `print(df[[...]].head())` prints a small sample of new features to confirm they exist and look sensible.

18. Hypothesis testing
   - A t test checks whether verified purchases have different star ratings than non verified purchases. The t statistic and p value are printed along with a short interpretation.

19. Bayesian probability example
   - Compute P(helpful | total_votes > 0) as the fraction of voted reviews that have helpful votes. This is printed as an example of an applied probability calculation.

Inputs and outputs

- Inputs: the CSV file with reviews.
- Outputs: a cleaned DataFrame `df` with added features, and multiple plots printed inline for exploration.

Notes and tips

- Keep the sample size manageable for presentations. Increase `nrows` only when you need production quality metrics.
- If `review_date` parsing fails in some rows, the `errors='coerce'` will convert them to NaT, so downstream code should handle missing dates.
- One hot encoding `product_category` may create many columns depending on categories. Consider grouping rare categories before encoding for model simplicity.

---

End of 04 Data analysis and preprocessing
