from d3m import index
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
import sys

# Creating pipeline
pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

# Step 1: Ts formatter
step_0 = PrimitiveStep(
    primitive=index.get_primitive(
        "d3m.primitives.data_preprocessing.data_cleaning.DistilTimeSeriesFormatter"
    )
)
step_0.add_argument(
    name="inputs", argument_type=ArgumentType.CONTAINER, data_reference="inputs.0"
)
step_0.add_output("produce")
pipeline_description.add_step(step_0)

# Step 1: dataset_to_dataframe on formatted dataset
step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_output('produce')
pipeline_description.add_step(step_1)

# Step 2: Grouping Field Compose
step_2 = PrimitiveStep(
    primitive=index.get_primitive(
        "d3m.primitives.data_transformation.grouping_field_compose.Common"
    )
)
step_2.add_argument(
    name="inputs",
    argument_type=ArgumentType.CONTAINER,
    data_reference="steps.1.produce",
)
step_2.add_output("produce")
pipeline_description.add_step(step_2)

# Step 3 DISTIL/NK Sloth primitive -> KMeans
step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.clustering.k_means.Sloth'))
step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
step_3.add_hyperparameter(name='nclusters', argument_type=ArgumentType.VALUE,data=3)
step_3.add_output('produce')
pipeline_description.add_step(step_3)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.3.produce')

# Output json pipeline
blob = pipeline_description.to_json()
filename = blob[8:44] + '.json'
with open(filename, 'w') as outfile:
    outfile.write(blob)
