import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Feature selection (independent variable)
    X = data['review']
    
    # Target variable (sentiment)
    y = data['sentiment'].map({'positive': 1, 'negative': 0})  # Convert labels to numeric
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
