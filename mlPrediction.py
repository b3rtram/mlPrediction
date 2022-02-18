import flask
import model
import numpy as np
import torch
import json
from sklearn.preprocessing import MinMaxScaler

app = flask.Flask(__name__)

myModel = model.Model()

@app.route("/")
def index():
    return "Hello World"


@app.route("/api/predict")
def predict():
    js = flask.request.json
    values = np.array(js["values"])

    scaler = MinMaxScaler(feature_range = (0, 1))
    values = scaler.fit_transform(values.reshape(-1,1))

    myModel.modelTrain(values)

    val = torch.FloatTensor(values)

    myModel.eval()
    p = myModel.forward(val.view(1,1,-1))
    a = p.detach().numpy() 
    print(a)
    inverse = scaler.inverse_transform(a)
    ret =  {"prediction": [inverse[0][0].astype(float)]}


    return json.dumps(ret)



