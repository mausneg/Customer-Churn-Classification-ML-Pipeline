import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
import os
from transform import CATEGORICAL_FEATURE_KEY, LABEL_KEY, NUMERIC_FEATURE_KEY, transformed_name

LABEL_KEY = 'Churn'


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

def model_builder():
    input_features = []
    for key in NUMERIC_FEATURE_KEY:
        input_features.append(tf.keras.layers.Input(shape=(1,), name=transformed_name(key)))
    for key, dim in CATEGORICAL_FEATURE_KEY.items():
        input_features.append(tf.keras.layers.Input(shape=(dim,), name=transformed_name(key)))
    inputs = tf.keras.layers.concatenate(input_features)
    print(inputs)  
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_features, outputs=outputs)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True)
    return model

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

def run_fn(fn_args: FnArgs) -> None:
    log_dir = os.path.join(fn_args.model_run_dir, 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        update_freq='batch'
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.01, 
        patience=20,
        restore_best_weights=True
    )
    plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        min_delta=0.01,
        patience=10,
        min_lr=0.001
    )
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_set = input_fn(fn_args.train_files, tf_transform_output, 20)
    val_set =  input_fn(fn_args.eval_files, tf_transform_output, 20)
    model = model_builder()
    model.fit(
        train_set,
        steps_per_epoch=fn_args.train_steps,
        validation_data=val_set,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback, early_stopping_callback, plateau_callback],
        epochs=20
    )
    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output,
        )
    }

    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
