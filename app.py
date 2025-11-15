from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 

#create flask app
app = Flask (__name__)

# load the pickle model
model = pickle.load(open('model.pkl','rb'))

print(type(model))
@app.route("/")
def index():
    return render_template('index.html')

categorical_columns = ['AgeGroup', 'PatientLifestyle',
       'FamilyHistory', 'NutritionalStatus', 'FBGOutcome',
         'RBGOutcome', 'ScreeningResult', 'Tested', 'TestResult','ScreenedFor']

@app.route("/predict", methods = ["POST"])
def predict():
  
    data = {
        'AgeGroup': request.form['ageGroup'],
        'Impotence':int (request.form['impotence']),
        'CVAStroke': int(request.form['cvaStroke']),
        'PatientLifestyle': request.form['patientLifestyle'],
        'FamilyHistory': request.form['familyHistory'],
        'BPDiastolic': float (request.form['bpDiastolic']),
        'BPSystolic': float (request.form['bpSystolic']),
        'BMI':float (request.form['bmi']),
        'NutritionalStatus': request.form['nutritionalStatus'],
        'FBGOutcome': request.form['fbgoutcome'],
        'RBGOutcome': request.form['rbgoutcome'],
        'ScreeningResult': request.form['screeningResult'],
        'ScreenedFor': request.form['screenedFor'],
        'DiagnosedWithHypertension': int (request.form['DiagnosedWithHypertension']),
        'Tested': request.form['tested'],
        'GlucoseFasting': float(request.form['glucoseFasting']),   
        'GlucoseRandom': float(request.form['glucoseRandom']),     
        'TestResult': request.form['testResult'],       
        'WithHypertension': int (request.form['withHypertension']),
        'HIVPositive': int(request.form['hivPositive']),

    }



    features = pd.DataFrame([data])

    for col in categorical_columns:
        le = LabelEncoder()
        features[col] = le.fit_transform (features[col])
  
    prediction = model.predict(features)
    
    if prediction[0] == 1 :
        result = "Diabetic"
    
    else:
        result = "Non-Diabetic"

    return render_template('index.html', prediction_text = f"You have Been Diagnossed {result}")

if __name__ == "__main__":
    app.run(debug=True)