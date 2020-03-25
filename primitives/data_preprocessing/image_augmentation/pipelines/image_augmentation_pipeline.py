
from d3m import index
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
import sys

# Create pipeline
pipeline_description = Pipeline()
pipeline_description.add_input(name = 'inputs')

# Step 0: Denormalize primitive
step_0 = PrimitiveStep(primitive = index.get_primitive('d3m.primitives.data_transformation.denormalize.Common'))
step_0.add_argument(name = 'inputs', argument_type = ArgumentType.CONTAINER, data_reference = 'inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

# Step 1: Dataset to Pandas DataFrame
step_1 = PrimitiveStep(primitive = index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
step_1.add_argument(name = 'inputs', argument_type = ArgumentType.CONTAINER, data_reference = 'steps.0.produce')
step_1.add_hyperparameter(name = 'dataframe_resource', argument_type = ArgumentType.VALUE, data = 'learningData')
step_1.add_output('produce')
pipeline_description.add_step(step_1)

# Step 2: Image augmentation
step_2 = PrimitiveStep(primitive = index.get_primitive('d3m.primitives.data_augmentation.image_augmentation.image_augmentation'))
step_2.add_argument(name = 'inputs', argument_type = ArgumentType.CONTAINER, data_reference = 'steps.1.produce')
step_2.add_hyperparameter(name = 'transform_group', argument_type = ArgumentType.VALUE, data = 'image_classification_option_1')
step_2.add_hyperparameter(name = 'data_path', argument_type = ArgumentType.VALUE, data = '/root/augmented_images/')
step_2.add_output('produce')
pipeline_description.add_step(step_2)

pipeline_description.add_output(name = 'output_predictions', data_reference = 'steps.2.produce')

# Output JSON pipeline
blob = pipeline_description.to_json()
filename = blob[8:44] + '.json'
with open(filename, 'w') as outfile:
    outfile.write(blob)