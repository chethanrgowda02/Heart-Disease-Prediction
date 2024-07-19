
# Heart Disease Prediction System

## Overview

This project is a web-based application that predicts the likelihood of heart disease based on user input. It uses a pre-trained Random Forest model to make predictions. The application is built using Flask, a lightweight web framework for Python.
#### Supervised by 
[Prof. Agughasi Victor Ikechukwu](https://github.com/Victor-Ikechukwu), 
(Assistant Professor) 
Department of CSE, MIT Mysore)

#### Collaborators
- 4MH21CS032 [Harsha S]
- 4MH21CS027 [Gautam Peabhu HM](https://github.com/Deepthik05)
- 4MH21CS025 [Gagan Gowda MS](https://github.com/nishanthkj)
- 4MH21CS014 [Chethan R]



## WebSite
[Heart Disease Prediction System](https://heart-disease-prediction-ef37.onrender.com/)

## Project Structure

```
Heart_Disease_Prediction/
│
├── models/
│   └── random_forest_model.h5
|   └── heart_disease_prediction_model.h5
│
├── templates/
│   └── index.html
│
├── app.py
└── model.py
```

## Files Description

- **models/random_forest_model.h5**: The pre-trained Random Forest model saved in HDF5 format.
- **templates/index.html**: The HTML template for the web form.
- **app.py**: The main Flask application file.
- **model.py**: Script used to train and save the Random Forest model.

## Setup Instructions

### Prerequisites

- Python 3.12
- Flask
- scikit-learn
- h5py
- numpy
- pandas

### Installation

1. **Clone the repository**:
    ```bash
    git clone 
    cd 
    ```

2. **Create a virtual environment**:
    ```bash
    # On Windows use
    python -m venv venv
    venv\Scripts\activate
    ```

3. **Install the required packages**:
    ```bash
    pip install flask scikit-learn h5py numpy pandas
    ```

4. **Ensure the model file exists**:
    Make sure `heart_disease_prediction_model.h5` is present in the `models` directory.

## Usage

1. **Run the Flask application**:
    ```bash
    python app.py
    ```

2. **Access the application**:
    Open your web browser and go to `http://127.0.0.1:5000/`.

3. **Enter the required details**:
    Fill in the form with the necessary details and click on the "Predict" button to get the prediction.

## Code Explanation

### `app.py`

This is the main Flask application file.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import h5py
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the model once when the application starts
with h5py.File('models/heart_disease_prediction_model.h5', 'r') as h5f:
    model_data = h5f['model'][()]
rf = pickle.loads(model_data)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve values from form
        features = [
            int(request.form['age']),
            int(request.form['sex']),
            int(request.form['cp']),
            int(request.form['trestbps']),
            int(request.form['chol']),
            int(request.form['fbs']),
            int(request.form['restecg']),
            float(request.form['thalach']),
            int(request.form['exang']),
            float(request.form['oldpeak']),
            int(request.form['slope']),
            int(request.form['ca']),
            int(request.form['thal'])
        ]

        # Make prediction
        prediction = rf.predict([features])

        # Interpret result
        result = "No Heart Disease" if prediction[0] == 0 else "Possibility of Heart Disease"
    except ValueError:
        result = "Please enter valid values in all fields"
    
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)


## Model Training (`model.py`)

import pandas as pd

# Load the dataset
dataset_path = 'data/dataset.csv'
data = pd.read_csv(dataset_path)

# Display the first few rows of the dataset to understand its structure
data.head()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import h5py
import numpy as np

# Split the dataset into features and target variable
X = data.drop(columns='target')
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest model
rf = RandomForestClassifier()

# Fit the model on the training data
rf.fit(X_train, y_train)

# Save the model to an HDF5 file
model_file_path = 'models/heart_disease_prediction_model.h5'
with h5py.File(model_file_path, 'w') as h5f:
    model_data = pickle.dumps(rf)
    h5f.create_dataset('model', data=np.void(model_data))

model_file_path

## Screenshots
![image](https://github.com/nishanthkj/Heart_Disease_Prediction/assets/138886231/64d81d3e-5c73-4cce-bea4-67baa38b12fa)

## Conclusion

This documentation provides a comprehensive guide to setting up and running the Heart Disease Prediction System. By following the steps outlined, you should be able to deploy the application and make predictions based on user input. If you encounter any issues, ensure that all dependencies are installed and that the model file is correctly placed in the `models` directory.
