from src.data_loader import load_data
from src.model import train_model
from src.evaluate import evaluate_model

def main():
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_data('data/movie_reviews.csv')
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
