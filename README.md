Project Report: Sentiment Analysis and Classification using RandomForest with Text Preprocessing
1. Introduction
The project focuses on performing sentiment analysis on healthcare reviews using Natural Language Processing (NLP) techniques. The pipeline includes text cleaning, sentiment analysis, feature extraction using TF-IDF, and classification using a RandomForest model. The primary objective is to classify the sentiment of reviews as positive, negative, or neutral based on their content.

2. Approach
2.1 Text Preprocessing
To ensure that the data is clean and usable for analysis, we performed the following steps:

Libraries Used:
nltk for stemming, lemmatization, and removal of stopwords.
re for text cleaning through regular expressions.
TextBlob for performing sentiment analysis.
TfidfVectorizer from sklearn for converting text data into numerical feature vectors.
Steps:
Lemmatization: The root form of each word is identified to ensure consistency (e.g., converting "running" to "run").
Stopwords Removal: Common words (e.g., "the", "is") that do not add significant meaning are removed.
Text Cleaning: Special characters, symbols, and numbers are removed using regular expressions.
2.2 Sentiment Analysis
Method: Sentiment analysis was performed using the TextBlob library. Each review's polarity is computed:
Positive polarity indicates a positive review.
Negative polarity indicates a negative review.
A neutral sentiment means the review does not lean towards either side.
2.3 TF-IDF Vectorization
TF-IDF (Term Frequency-Inverse Document Frequency):
The cleaned text data was converted into feature vectors using TfidfVectorizer, which considers both the frequency of words and their importance relative to the entire corpus.
3. Classification Model: RandomForest
3.1 Model Building
RandomForest Classifier:
The RandomForestClassifier from sklearn was chosen for its ability to handle high-dimensional data and prevent overfitting.
Model Parameters: A simple RandomForest model was used initially with a max_depth=5.
3.2 Cross-Validation
Purpose: To evaluate the model’s performance and prevent overfitting, we applied 5-fold cross-validation.
Results: The cross-validation accuracy scores were computed, giving a mean accuracy score across the folds.
3.3 Hyperparameter Tuning
Grid Search: We performed a GridSearchCV to optimize hyperparameters such as:
Number of trees (n_estimators),
Maximum depth (max_depth),
Minimum samples required at a leaf node (min_samples_leaf).
Best Parameters: The grid search yielded the best hyperparameters, improving the overall model performance.
4. Results
4.1 Cross-Validation Scores
RandomForest Classifier:
Cross-validation scores varied slightly between folds, but the overall mean cross-validation accuracy was stable, indicating good model performance.
Mean Accuracy: Displayed after cross-validation, showing consistent performance.
4.2 Classification Report
Metrics: For the test set, we evaluated the model using the following metrics:
Precision: How many selected items are relevant.
Recall: How many relevant items are selected.
F1-Score: Harmonic mean of precision and recall.
Confusion Matrix: Showed the true positives, false positives, true negatives, and false negatives for the classification.
4.3 Hyperparameter Tuning Results
Best Parameters: After performing GridSearchCV, the following were the optimal hyperparameters:
n_estimators=200
max_depth=10
min_samples_leaf=2
Classification Performance: With tuned parameters, the model showed significant improvements in classification accuracy, precision, recall, and F1-score.
5. Conclusion
The RandomForest model was effective in classifying the sentiment of healthcare reviews after applying NLP preprocessing techniques and TF-IDF vectorization. Cross-validation and hyperparameter tuning further optimized the model's performance.
