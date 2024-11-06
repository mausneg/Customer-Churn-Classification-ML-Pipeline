import pandas as pd
import tensorflow as tf
import requests
import json
import base64

def string_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[bytes(value, "utf-8")]
        ),
    )
    
def float_feature(value):
    return tf.train.Feature(
        float_list=tf.train.FloatList(
            value=[value]
        ),
    )
    
def int_feature(value):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(
            value=[value]
        ),
    )

def _tensor_to_dict(inputs):
    outputs = {}
    for key, value in inputs.items():
        if isinstance(value, tf.Tensor):
            outputs[key] = value.numpy().flatten().tolist()
        else:
            outputs[key] = value
    return outputs


def prepare_json(inputs):
    inputs = _tensor_to_dict(inputs)
    feature_spec = dict()
    for keys, values in inputs.items():
        if isinstance(values, float):
            feature_spec[keys] = float_feature(values)
        elif isinstance(values, int):
            feature_spec[keys] = int_feature(values)
        elif isinstance(values, str):
            feature_spec[keys] = string_feature(values)
    example = tf.train.Example(
        features=tf.train.Features(feature=feature_spec)
    ).SerializeToString()
    result = [
        {
            "examples": {
                "b64": base64.b64encode(example).decode()
            }
        }
    ]
    return json.dumps({
        "signature_name": "serving_default",
        "instances": result,
    })

def predict(inputs):
    url = "http://localhost:8080/v1/models/customer-churn-model:predict"
    headers = {"content-type": "application/json"}
    data = []
    for i in range(len(inputs['Age'])):
        data.append(prepare_json({key: value[i] for key, value in inputs.items()}))
    
    responses = []
    for i in range(len(inputs['Age'])):
        response = requests.post(url, data=data[i], headers=headers)
        response = 'Churn' if response.json()['predictions'][0][0] > 0.5 else 'Not Churn'
        responses.append(response)
    return responses

if __name__ == '__main__':
    df_data = pd.read_csv("data/val.csv")
    df_data.pop("Churn")
    df_data = df_data.sample(5)

    inputs = {key: value for key, value in df_data.to_dict(orient="list").items()}
    prediction = predict(inputs)
    print(prediction)