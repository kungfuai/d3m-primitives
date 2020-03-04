from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

# Creating pipeline
pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

# Step 0: Denormalize dataset resources
step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.denormalize.Common'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

# Step 1: Dataset sample primitive to reduce computation time
step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_preprocessing.dataset_sample.Common'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_hyperparameter(name='sample_size', argument_type= ArgumentType.VALUE, data=100)
step_1.add_output('produce')
pipeline_description.add_step(step_1)

# Step 2: dataset_to_dataframe
step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_2.add_hyperparameter(name='dataframe_resource', argument_type= ArgumentType.VALUE, data='learningData')
step_2.add_output('produce')
pipeline_description.add_step(step_2)

# Step 3: column parser on input DF
step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.Common'))
step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
step_3.add_output('produce')
step_3.add_hyperparameter(name='parse_semantic_types', argument_type= ArgumentType.VALUE, data=["http://schema.org/Boolean",
    "http://schema.org/Integer",
    "http://schema.org/Float",
    "https://metadata.datadrivendiscovery.org/types/FloatVector"])
pipeline_description.add_step(step_3)

# Step 4: parse attribute semantic types 
step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
step_4.add_hyperparameter(name='semantic_types', argument_type= ArgumentType.VALUE, data=["https://metadata.datadrivendiscovery.org/types/Attribute"])
step_4.add_output('produce')
pipeline_description.add_step(step_4)

# Step 5: parse target semantic types
step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
step_5.add_hyperparameter(name='semantic_types', argument_type= ArgumentType.VALUE, data=["https://metadata.datadrivendiscovery.org/types/Target",
    "https://metadata.datadrivendiscovery.org/types/TrueTarget",
    "https://metadata.datadrivendiscovery.org/types/SuggestedTarget"])
step_5.add_output('produce')
pipeline_description.add_step(step_5)

# Step 6: Gator primitive
step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.classification.inceptionV3_image_feature.Gator'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_6.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_6.add_hyperparameter(name='unfreeze_proportions', argument_type=ArgumentType.VALUE, data=[0.5])
# step_6.add_hyperparameter(name='top_layer_epochs', argument_type=ArgumentType.VALUE, data=1)
# step_6.add_hyperparameter(name='all_layer_epochs', argument_type=ArgumentType.VALUE, data=1)
step_6.add_output('produce')
pipeline_description.add_step(step_6)

# Step 7: construct predictions
step_7 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.6.produce')
step_7.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
step_7.add_output('produce')
pipeline_description.add_step(step_7)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.7.produce')

# Output json pipeline
blob = pipeline_description.to_json()
filename = blob[8:44] + '.json'
#filename = "pipeline.json"
with open(filename, 'w') as outfile:
    outfile.write(blob)