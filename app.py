from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained supply chain model
model_path = 'supplychain.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array([int_features])  # Convert to 2D array for model

    # Make prediction
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)  # Rounding the predicted supply rate

    return render_template('index.html', prediction_text=f'Supply Prediction Rate: {output}')

if __name__ == "__main__":
    app.run(debug=True)
