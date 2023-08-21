import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the past lottery data (replace 'lottery_data.csv' with your dataset)
def load_lottery_data(file_path):
    return pd.read_csv(file_path)

# Split data into features (X) and target (y)
def get_features_target(data):
    X = data.drop('Winner', axis=1)
    y = data['Winner']
    return X, y

# Train a Decision Tree Classifier
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

if __name__ == "__main__":
    # Load the data (replace 'lottery_data.csv' with your dataset)
    data = load_lottery_data('lottery_data.csv')

    # Split the data into features (X) and target (y)
    X, y = get_features_target(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Decision Tree model
    model = train_decision_tree(X_train, y_train)

    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    print("Model accuracy:", accuracy)
