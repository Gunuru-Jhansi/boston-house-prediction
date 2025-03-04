import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])  # Fixed 'method' to 'methods'
def predict_api():
    data = request.get_json()['data']  # Corrected to get_json()
    print(data)
    input_data = np.array(list(data.values())).reshape(1, -1)
    print(input_data)
    new_data = scalar.transform(input_data)
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The house price prediciton is {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)