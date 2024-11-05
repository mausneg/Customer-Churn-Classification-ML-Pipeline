import pandas as pd
import tensorflow as tf
import tensorflow_transform as tft
import requests
import json
import base64

NUMERIC_FEATURE_KEY = ['Age', 'Last Interaction', 'Payment Delay', 'Support Calls', 'Tenure', 'Usage Frequency']
CATEGORICAL_FEATURE_KEY = {
    'Contract Length': 3, 
    'Subscription Type': 3, 
    'Gender': 2
}
LABEL_KEY = 'Churn'

TRANSFORM_ARTIFACTS_DIR = 'pipelines/churn-pipeline/Transform/transform_graph/5'

def _load_transform_artifacts():
    return tft.TFTransformOutput(TRANSFORM_ARTIFACTS_DIR) 

@tf.function
def _convert_sparse_to_dense(sparse_tensor):
    dense_tensor = {}
    for key, value in sparse_tensor.items():
        if isinstance(value, tf.SparseTensor):
            dense_tensor[key] = tf.sparse.to_dense(sparse_tensor[key])
        else:
            dense_tensor[key] = sparse_tensor[key]
    return dense_tensor
    

def transformed_name(key):
    return key.replace(' ', '_').lower() + '_xf'

def preprocessing_fn(inputs, transform_output):
    outputs = {}
    for key in NUMERIC_FEATURE_KEY:
        outputs[transformed_name(key)] = transform_output.transform_raw_features({key: inputs[key]})[transformed_name(key)]
    for key, value in CATEGORICAL_FEATURE_KEY.items():
        outputs[transformed_name(key)] = transform_output.transform_raw_features({key: inputs[key]})[transformed_name(key)]
    return outputs

def _to_sparse_tensor(df,column_name):
    indices = []
    values = []

    for i, value in enumerate(df[column_name]):
        if pd.notna(value):
            indices.append([i, 0])
            values.append(value)
    dtype = tf.string if df[column_name].dtype == object else tf.int64

    sparse_tensor = tf.sparse.SparseTensor(
        indices=indices,
        values=tf.constant(values, dtype=dtype),
        dense_shape=[len(df), 1]
    )
    return sparse_tensor

def get_model_metadata():
    url = "http://localhost:8080/v1/models/customer-churn-model/metadata"
    headers = {"content-type": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        metadata = response.json()
        print(json.dumps(metadata, indent=2))  # Pretty-print JSON metadata
        return metadata
    else:
        print(f"Failed to get metadata: {response.status_code}")
        return None  

def serialize_example(inputs):
    feature = {}
    for key, value in inputs.items():
        if isinstance(value[0], float):
            feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
        elif isinstance(value[0], int):
            feature[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        else:
            feature[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(v).encode('utf-8') for v in value]))
    
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def predict(inputs):
    serialized_example = serialize_example(inputs)
    serialized_inputs = [base64.b64encode(serialized_example).decode('utf-8')]
    json_data = {'signature_name': 'serving_default', 'instances': [{'examples': serialized_inputs[0]}]}
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8080/v1/models/customer-churn-model:predict', data=json.dumps(json_data), headers=headers)
    predictions = json.loads(json_response.text).get('predictions', None)
    return predictions

if __name__ == '__main__':
    df_data = pd.read_csv('data/val.csv')
    df_data = df_data.sample(5)
    sparse_tensor_dict = {col: _to_sparse_tensor(df_data, col) for col in df_data.columns}
    transform_output = _load_transform_artifacts()
    transformed_features = preprocessing_fn(sparse_tensor_dict, transform_output)
    dense_inputs = _convert_sparse_to_dense(transformed_features)
    predictions = predict(dense_inputs)
    print(predictions)