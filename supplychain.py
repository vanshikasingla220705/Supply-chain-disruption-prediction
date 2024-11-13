# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
data = pd.read_csv("supply_disruption_prediction_data.csv")

# Define the features and target variable
X = data[['PoliticalEventImpact', 'MarketTrend', 'WeatherCondition']]
y = data['SupplyPredictionRate']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a .pkl file without using a dictionary
with open("supplychain.pkl", 'wb') as file:
    pickle.dump(model, file)

print("Model saved to 'supplychain.pkl'")
