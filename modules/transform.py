import tensorflow as tf
import tensorflow_transform as tft

NUMERIC_FEATURE_KEY = ['Age', 'Last Interaction', 'Payment Delay', 'Support Calls', 'Tenure', 'Usage Frequency']
CATEGORICAL_FEATURE_KEY = {
    'Contract Length': 3, 
    'Subscription Type': 3, 
    'Gender': 2
}
LABEL_KEY = 'Churn'
        
def transformed_name(key):
    return key + '_xf'


def preprocessing_fn(inputs):
    outputs = {}
    for key in NUMERIC_FEATURE_KEY:
        outputs[transformed_name(key)] = tft.scale_to_0_1(inputs[key])

    for key, value in CATEGORICAL_FEATURE_KEY.items():
        integerized = tft.compute_and_apply_vocabulary(inputs[key], vocab_filename=key)
        integerized_dense = tf.sparse.to_dense(integerized, default_value=0)
        outputs[transformed_name(key)] = tf.reshape(tf.one_hot(integerized_dense, value), [-1, value])
              
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    return outputs