from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd 

import config
import predict

app = Flask(__name__)

# Testing path
@app.route('/', methods=['GET'])
def saludar():
    return jsonify({'mensaje': 'Hola desde el Server!!!'})

@app.route('/predict', methods=['POST'])
def predictAPI_v1():
    json_data = request.get_json()
    json_data = json.dumps(json_data)
    json_data = json.loads(json_data)
    dataframe = pd.DataFrame.from_dict(json_data, orient="index")
    resultado = predict.make_predictions(dataframe)
    return jsonify({'mensaje': resultado[0]})

if(__name__ == '__main__'):
    app.run(debug=True)
