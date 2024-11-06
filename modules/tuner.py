from typing import Any, Dict, NamedTuple, Text
import keras_tuner as kt
import tensorflow as tf
from keras_tuner.engine import base_tuner
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
import os
from transform import CATEGORICAL_FEATURE_KEY, NUMERIC_FEATURE_KEY, LABEL_KEY, transformed_name

EPOCHS = 50
BATCH_SIZE = 512

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames=filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epocs, batch_size=64)->tf.data.Dataset:
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    def _to_dense(features, labels):
        if isinstance(labels, tf.sparse.SparseTensor):
            labels = tf.sparse.to_dense(labels, default_value=0)
        return features, labels

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epocs,
        label_key=transformed_name(LABEL_KEY)
    )
    dataset = dataset.map(_to_dense)
    return dataset

def build_model(hp):
    inputs = []
    for key in NUMERIC_FEATURE_KEY:
        inputs.append(tf.keras.layers.Input(shape=(1,), name=transformed_name(key)))

    for key, dim in CATEGORICAL_FEATURE_KEY.items():
        inputs.append(tf.keras.layers.Input(shape=(dim,), name=transformed_name(key)))

    x = tf.keras.layers.concatenate(inputs)
    for i in range(hp.Int('num_layers', 1, 4)):
        x = tf.keras.layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32), activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(rate=hp.Float('dropout_' + str(i), 0.2, 0.5, step=0.1))(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    return model


def tuner_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, EPOCHS, BATCH_SIZE)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, EPOCHS, BATCH_SIZE)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.01, 
        patience=40,
        restore_best_weights=True
    )

    tuner = kt.Hyperband(
        hypermodel=build_model,
        objective=kt.Objective('val_binary_accuracy', direction='max'),
        max_epochs=EPOCHS,
        factor=3,
        directory=fn_args.working_dir,
        project_name='churn',
    )

    tuner.oracle.max_trials = 20

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_data': eval_dataset,
            'validation_steps': fn_args.eval_steps,
            'callbacks': [
                early_stopping_callback, 
            ]
        }
    )
