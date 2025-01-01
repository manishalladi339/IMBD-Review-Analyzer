from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def train_model(X_train, y_train):
    # Create a pipeline with TfidfVectorizer and Logistic Regression
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model.fit(X_train, y_train)
    return model
