# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import pickle
from flask import Flask,request,jsonify,render_template,url_for
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('lrmodel.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=np.array(list(data.values())).reshape(1,-1)
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = [x for x in request.form.values()]
    print(data)

    # Define a mapping for string values to numerical values
    value_mapping = {
        "Financial": 0.0,
        "Food":1.0,
        "Sanity":2.0,
        "Teachers":3.0,
        "Transportation":4.0,
        "Gen": 0.0,
        "SC":1.0,
        "ST":2.0,
        "OBC":3.0,
        "Other":4.0
        # Add more mappings as needed
    }

    # Check if the values in data can be mapped, otherwise, keep them as is
    for i in range(len(data)):
        if data[i] in value_mapping:
            data[i] = value_mapping[data[i]]

    final_input = np.array(data, dtype=float).reshape(1, -1)
    print(final_input)

    output = model.predict(final_input)[0]
    return render_template("home.html", prediction_text="Scheme : {}".format(output))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
