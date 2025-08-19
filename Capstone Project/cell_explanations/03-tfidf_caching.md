# 03 TF-IDF caching and artifact management

Overview

This cell builds or reuses a TF-IDF vectorizer and the corresponding document term matrix. The artifacts are saved to disk so repeated runs do not need to recompute expensive text feature extraction.

Purpose

- Create a TF-IDF vectorizer from review text.
- Save the fitted vectorizer and the matrix to disk.
- Reuse saved artifacts when available.
- Adjust sample sizes automatically when `SMOKE_TEST` is active.

Line by line explanation

1. `from sklearn.feature_extraction.text import TfidfVectorizer`
   - Import the TF-IDF transformer that converts raw text into weighted term frequency features.

2. `import pandas as pd`
   - Import pandas for reading the CSV and manipulating the sample of texts.

3. `TFIDF_PATH = ARTIFACTS / "tfidf.pkl"`
   - Path where the fitted `TfidfVectorizer` object will be saved.

4. `TFIDF_MATRIX = ARTIFACTS / "tfidf_matrix.pkl"`
   - Path where the TF-IDF document matrix will be saved.

5. `DATA_CSV = "amazon_reviews_us_Apparel_v1_00.csv"`
   - A variable holding the name of the CSV file that contains review texts.

6. `def _get_reviews_sample(n=20000 if not SMOKE_TEST else 2000):` and body
   - Helper that reads `n` rows from the CSV, drops rows without review text, and returns the review strings list. The default `n` depends on `SMOKE_TEST`.

7. `if already(TFIDF_PATH) and already(TFIDF_MATRIX):`
   - If both artifacts exist and are non-empty, reuse them to save time.

8. `    tfidf = load_pickle(TFIDF_PATH)` and `X_tfidf = load_pickle(TFIDF_MATRIX)`
   - Load the saved vectorizer and matrix from the `artifacts` folder.

9. `else:`
   - If artifacts are missing, create them from the CSV.

10. `    texts = _get_reviews_sample()`
    - Read a sample of review texts using the helper defined earlier.

11. `    tfidf = TfidfVectorizer(max_features=50000 if not SMOKE_TEST else 5000)`
    - Create a `TfidfVectorizer`. The maximum number of features is smaller in smoke test mode to reduce memory and compute time.

12. `    X_tfidf = tfidf.fit_transform(texts)`
    - Fit the vectorizer to the sample texts and transform them into a sparse matrix of TF-IDF features.

13. `    save_pickle(tfidf, TFIDF_PATH)` and `save_pickle(X_tfidf, TFIDF_MATRIX)`
    - Persist the fitted vectorizer and the computed matrix to disk for reuse in later runs.

14. `    print("Saved TF-IDF artifacts.")`
    - Inform the user that artifacts were created and saved.

Inputs and outputs

- Inputs: The CSV file with reviews and the `SMOKE_TEST` flag.
- Outputs: Pickle files in `artifacts/` containing the vectorizer and TF-IDF matrix, or reuse of those files.

Notes and tips

- The TF-IDF matrix can be large in memory. If you run into memory errors, reduce `max_features` or reduce the sample size.
- Because the saved `TfidfVectorizer` carries vocabulary and IDF weights, reusing it ensures consistent feature mapping across experiments.
- If you change vectorizer parameters such as `max_features` or stop words, delete the old artifacts to force recomputation.

---

End of 03 TF-IDF caching and artifact management
