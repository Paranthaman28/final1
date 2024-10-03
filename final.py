import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  #regularization
from textblob import TextBlob
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer



##steming --> indentify the root word
stemmer=PorterStemmer()

Lemmatizer=WordNetLemmatizer()

def clean_text(sentence1):
    corpu=[]
    for i in range(len(sentence1)):
        review=re.sub('[^a-zA-Z]',' ',sentence1[i])#the line explain that sentence which remove special letter or symbol other then the alphapet
        review=review.lower()
        review=review.split()
        review=[Lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
        review=' '.join(review)
        corpu.append(review)
    return corpu

def get_sentiment(corpu):
        sentiments=[]
        for my_texts in corpu:
            anlaysis=TextBlob(my_texts)
            if anlaysis.sentiment.polarity>0:
                sentiments.append('positive')
            elif anlaysis.sentiment.polarity<0:
                sentiments.append('negative')
            else:
                sentiments.append('neutral')
        return sentiments
df=pd.read_csv(r"C:\Users\Paranthaman\Downloads\synthetic_healthcare_reviews_with_text.csv")
df.fillna(method='ffill', inplace=True) 
df['clear_review']=clean_text(df['Review_Text'].values)

df['sentiment']=get_sentiment(df['clear_review'])

X=df['clear_review']
y=df['sentiment']

tfidf=TfidfVectorizer()
X=tfidf.fit_transform(X)

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def model(models):##cross validation
    models=RandomForestClassifier(max_depth=5, random_state=42)
    models.fit(X_train,y_train)
    # Cross-validation to check for overfitting
    cv_scores = cross_val_score(models, X_train, y_train, cv=5, scoring='accuracy')
    print(f"***{type(models).__name__}***")
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean()}")
    return cv_scores

from sklearn.model_selection import train_test_split, GridSearchCV
def grid(rf):
    param_grid_rf = {
        'n_estimators': [50, 100, 200],  # Number of trees
        'max_depth': [5, 10, 15],  # Limit the depth of the tree to avoid overfitting
        'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required at a leaf node
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
    grid_search_rf.fit(X_train, y_train)

    print("Best Parameters for Random Forest: ", grid_search_rf.best_params_)
    y_pred_rf = grid_search_rf.predict(X_test)
    print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
    return y_pred_rf