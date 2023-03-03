"""
This Python script is responsible for training the text classifier for 
predicting reliability of a news article.

The general approach taken is as follows:
1. preprocess the articles `text` using:
    a. Bag of Words (Count Vectorizer)
    b. TF-IDF
2. train an SVM using L2 regularization
3. model is pickled for future inference
"""

from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from utils.data_loader import data_loader


def train():
    """
    Function responsible for training the ML model and storing the artifact
    binary for later use.
    """
    # load data
    train_df = data_loader("./data/fake_news/clean_train.csv")

    # pipeline
    model = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None)),
    ])

    model.fit(train_df.text, train_df.label)

    # metrics
    preds = model.predict(train_df.text)
    print(metrics.classification_report(train_df.label, preds))

    # save model
    dump(model, 'model.joblib')


if __name__ == "__main__":
    train()
