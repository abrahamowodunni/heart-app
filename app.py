import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import numpy as np
app = Flask(__name__, template_folder='template')

# Load the pre-trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = StandardScaler()

def scale_inputs(inputs):
    # Scale the input features using your scaler if needed
    scaled_inputs = scaler.fit_transform([inputs])
    # Return the scaled inputs
    return inputs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    features = [
        int(request.form['BMI']),
        int(request.form['Smoking']),
        int(request.form['AlcoholDrinking']),
        int(request.form['Stroke']),
        float(request.form['PhysicalHealth']),
        float(request.form['MentalHealth']),
        int(request.form['DiffWalking']),
        int(request.form['Sex']),
        int(request.form['AgeCategory']),
        int(request.form['Race']),
        int(request.form['Diabetic']),
        int(request.form['PhysicalActivity']),
        int(request.form['GenHealth']),
        float(request.form['SleepTime']),
        int(request.form['Asthma']),
        int(request.form['KidneyDisease']),
        int(request.form['SkinCancer'])
    ]

    #prediction_ = model.predict(features)

    # Scale the input features
    #scaled_features = scaler.fit([features])

    # Make predictions
    #raw_probability = model.predict(scaled_features)
    #confidence = raw_probability * 100  # Convert to percentage

    # Render the result template
    #return render_template('result.html', result=raw_probability)
    # Convert the features list to an np.array
    features_array = np.array(features)

    # Scale the input features
    scaled_features = scale_inputs(features_array)

    # Make predictions
    raw_prediction = model.predict([scaled_features])

    # Render the result template or handle the prediction as needed
    return render_template('result.html', result=raw_prediction)

if __name__ == '__main__':
    app.run(debug=True)
