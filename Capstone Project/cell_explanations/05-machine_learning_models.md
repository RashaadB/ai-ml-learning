# 05 Machine learning models

Overview

This cell builds features, sets up simple TF-IDF features, creates targets for regression and classification, trains a broad set of baseline models, and evaluates them using standard metrics. It also demonstrates common techniques to handle class imbalance.

Purpose

- Engineer features used by models.
- Fit several regression models to predict star rating.
- Fit multiple classifiers to predict sentiment class.
- Demonstrate imbalance handling with SMOTE, under sampling, and class weights.
- Show simple hyperparameter tuning with grid search.

Line by line explanation

1. `# Section 2: Machine Learning Models` and imports
   - The cell imports a long list of tools from scikit learn and from imbalanced learn and plotting libraries. These tools support model building, evaluation, and visualization.

2. Load and clean data
   - `df = pd.read_csv(..., nrows=50000)` reads in data and `df.dropna(...)` ensures rows used have the fields needed for models.

3. Feature engineering
   - `review_length`: number of words in the review.
   - `sentiment`: a polarity score from TextBlob which ranges roughly from -1 to 1.
   - `helpfulness_score`: helpful votes divided by total votes with a safe fallback of 0 to avoid division by zero.
   - `verified_purchase_binary` and `vine_binary`: convert categorical flags into numeric indicators.
   - `sentiment_class`: bucket sentiment into `positive`, `negative`, or `neutral` using thresholds at 0.1 and -0.1.

4. TF-IDF features for text
   - `TfidfVectorizer(max_features=100)` extracts the top 100 textual features. The matrix is converted to a DataFrame and concatenated with the main `df` so text features are available as columns.

5. Normalization
   - Apply `MinMaxScaler` to scale `total_votes`, `helpful_votes`, and `review_length` to the [0, 1] range for models that are sensitive to feature scale.

6. Regression setup and evaluation utility
   - `X_reg` and `y_reg` are defined for predicting `star_rating`.
   - `eval_reg` is a helper function that prints R2, MSE, and RMSE for a trained model.

7. Fit baseline regression models
   - Fit and evaluate Linear Regression, Ridge, and Lasso and print results.

8. Visualization for model context
   - Boxplots, word cloud, scatter plots and correlation heatmaps provide visual context about data distribution and relationships used by models.

9. Classification setup
   - Select a small set of features as `X_clf` and convert the sentiment labels into integer form with `LabelEncoder`.
   - Split into train and test sets with a fixed random seed for reproducibility.

10. Train multiple classifiers
   - Logistic Regression, Decision Tree, Random Forest, Naive Bayes, and KNN are trained and evaluated using the classification report which shows precision, recall, and F1 score per class.

11. Techniques for imbalance
   - SMOTE: generate synthetic minority samples and evaluate a logistic regression trained on the balanced data.
   - Undersampling: randomly remove samples from the majority class to balance and evaluate.
   - Class weights: compute balanced class weights and pass them to LogisticRegression to bias learning toward minority classes.

12. Ensemble models
   - Train Random Forest, AdaBoost, and Gradient Boosting and print evaluation reports for each.
   - Build a stacking classifier using multiple base models and a logistic regression meta classifier. Train and evaluate the stacked model.

13. Grid search
   - Define a small hyperparameter grid for RandomForest and run `GridSearchCV` with 3 fold cross validation and f1 macro scoring. Report the best parameters and evaluate the best estimator on the test set.

Inputs and outputs

- Inputs: cleaned `df` from the earlier cell.
- Outputs: trained model objects in memory, printed metrics and plots.

Notes and tips

- The TF-IDF matrix is converted to dense arrays to concatenate with `df` and may use considerable memory. For larger datasets use sparse handling or feature selection.
- Hyperparameter tuning can be time consuming. Use smaller subsets and `SMOKE_TEST` when demoing.
- Save models you want to keep to disk using `pickle` or a model specific save function before running additional experiments that may overwrite memory.

---

End of 05 Machine learning models
