import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
import os
from transform import CATEGORICAL_FEATURE_KEY, LABEL_KEY, NUMERIC_FEATURE_KEY, transformed_name
from tuner import input_fn

LABEL_KEY = 'Churn'
BATCH_SIZE = 512
EPOCHS = 50


def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name="examples"),
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
        """Return the output to be used in the serving signature."""

        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec,
        )

        transformed_features = model.tft_layer(parsed_features)
        outputs = model(transformed_features)

        return {"outputs": outputs}

    return serve_tf_examples_fn

def model_builder(hp):
    inputs = []
    for key in NUMERIC_FEATURE_KEY:
        inputs.append(tf.keras.layers.Input(shape=(1,), name=transformed_name(key)))
    for key, dim in CATEGORICAL_FEATURE_KEY.items():
        inputs.append(tf.keras.layers.Input(shape=(dim,), name=transformed_name(key)))
    x = tf.keras.layers.concatenate(inputs)
    for i in range(hp['num_layers']):
        x = tf.keras.layers.Dense(units=hp['units_' + str(i)], activation='relu')(x)
        x = tf.keras.layers.Dropout(rate=hp['dropout_' + str(i)])(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp['learning_rate']),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    return model

def run_fn(fn_args: FnArgs):
    hp = fn_args.hyperparameters['values']
    log_dir = os.path.join(fn_args.model_run_dir, 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        update_freq='batch'
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.01, 
        patience=40,
        restore_best_weights=True
    )
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files, tf_transform_output, num_epocs=EPOCHS, batch_size=BATCH_SIZE)
    val_set =  input_fn(fn_args.eval_files, tf_transform_output, num_epocs=EPOCHS, batch_size=BATCH_SIZE)
    model = model_builder(hp)
    model.fit(
        train_set,
        steps_per_epoch=fn_args.train_steps,
        validation_data=val_set,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback, early_stopping_callback],
        epochs=hp['tuner/epochs'],
    )
    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                                    tf.TensorSpec(
                                    shape=[None],
                                    dtype=tf.string,
                                    name='examples'))
    }

    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
