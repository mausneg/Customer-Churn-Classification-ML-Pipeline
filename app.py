import os
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Tuner, Evaluator, Pusher
from tfx.proto import example_gen_pb2, trainer_pb2, transform_pb2, pusher_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

PIPELINE_NAME = 'mausneg-pipeline'
SCHEMA_PIPELINE_NAME = 'churn-tfdv-schema'

PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')
SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)

DATA_ROOT = 'data'

context = InteractiveContext(pipeline_root=PIPELINE_ROOT)

input = example_gen_pb2.Input(splits=[
    example_gen_pb2.Input.Split(name='train', pattern='*training*'),
    example_gen_pb2.Input.Split(name='eval', pattern='*testing*')
])
example_gen = CsvExampleGen(input_base=DATA_ROOT, input_config=input)

context.run(example_gen)

statistics_gen = StatisticsGen(
    examples=example_gen.outputs['examples']
)

context.run(statistics_gen)
context.show(statistics_gen.outputs['statistics'])

schema_gen = SchemaGen(
    statistics=statistics_gen.outputs['statistics']
)

context.run(schema_gen)
context.show(schema_gen.outputs['schema'])

example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema']
)

context.run(example_validator)
context.show(example_validator.outputs['anomalies'])

transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(os.path.join('modules', 'transform.py')),
)

context.run(transform)

tuner = Tuner(
    module_file=os.path.abspath(os.path.join('modules', 'tuner.py')),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],   
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=8000),
    eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=6000),
)

context.run(tuner)

trainer = Trainer(
    module_file=os.path.abspath(os.path.join('modules', 'trainer.py')),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    hyperparameters=tuner.outputs['best_hyperparameters'],
    train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=8000),
    eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=6000)
)

context.run(trainer)

model_resolver = Resolver(
    strategy_class= LatestBlessedModelStrategy,
    model = Channel(type=Model),
    model_blessing = Channel(type=ModelBlessing)
).with_id('Latest_blessed_model_resolver')
 
context.run(model_resolver)

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='Churn')],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(class_name='ExampleCount'),
            tfma.MetricConfig(class_name='AUC'),
            tfma.MetricConfig(class_name='FalsePositives'),
            tfma.MetricConfig(class_name='TruePositives'),
            tfma.MetricConfig(class_name='FalseNegatives'),
            tfma.MetricConfig(class_name='TrueNegatives'),
            tfma.MetricConfig(class_name='BinaryAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value':0.5}),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value':0.0001})
                    )
            )
        ])
    ]
)

evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config)
 
context.run(evaluator)

eval_result = evaluator.outputs['evaluation'].get()[0].uri
tfma_result = tfma.load_eval_result(eval_result)
tfma.view.render_slicing_metrics(tfma_result)
tfma.addons.fairness.view.widget_view.render_fairness_indicator(
    tfma_result
)

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory='serving_model_dir/customer-churn-model'
        )
    )
)
context.run(pusher)
context.show(pusher.outputs['pushed_model'])