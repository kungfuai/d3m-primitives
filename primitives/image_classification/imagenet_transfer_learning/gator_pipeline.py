from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

from primitives.pipeline_base import PipelineBase

class GatorPipeline(PipelineBase):

    def __init__(
        self,
        sample_size: int = 1000,
        all_layer_epochs: int = 100,
        top_layer_epochs: int = 100
    ):

        pipeline_description = Pipeline()
        pipeline_description.add_input(name="inputs")

        # Denormalize primitive
        step = PrimitiveStep(
            primitive = index.get_primitive(
                'd3m.primitives.data_transformation.denormalize.Common'
            )
        )
        step.add_argument(
            name = 'inputs', argument_type = ArgumentType.CONTAINER, data_reference = 'inputs.0'
        )
        step.add_output('produce')
        pipeline_description.add_step(step)

        # Dataset sample primitive to reduce computation time
        step = PrimitiveStep(
            primitive=index.get_primitive(
                'd3m.primitives.data_preprocessing.dataset_sample.Common'
            )
        )
        step.add_argument(
            name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce'
        )
        step.add_hyperparameter(
            name='sample_size', argument_type= ArgumentType.VALUE, data=sample_size
        )
        step.add_output('produce')
        pipeline_description.add_step(step)

        # DS to DF
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_transformation.dataset_to_dataframe.Common"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.1.produce",
        )
        step.add_hyperparameter(
            name='dataframe_resource', argument_type= ArgumentType.VALUE, data='learningData'
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # column parser on input DF
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_transformation.column_parser.Common"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.2.produce",
        )
        step.add_output("produce")
        step.add_hyperparameter(
            name="parse_semantic_types",
            argument_type=ArgumentType.VALUE,
            data=[
                "http://schema.org/Boolean",
                "http://schema.org/Integer",
                "http://schema.org/Float",
                "https://metadata.datadrivendiscovery.org/types/FloatVector",
            ],
        )
        pipeline_description.add_step(step)

        # parse attribute semantic types
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.3.produce",
        )
        step.add_hyperparameter(
            name="semantic_types",
            argument_type=ArgumentType.VALUE,
            data=[
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            ],
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # parse target semantic types
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.3.produce",
        )
        step.add_hyperparameter(
            name="semantic_types",
            argument_type=ArgumentType.VALUE,
            data=[
                "https://metadata.datadrivendiscovery.org/types/Target",
            ],
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # Gator
        step = PrimitiveStep(
            primitive=index.get_primitive(
                'd3m.primitives.classification.inceptionV3_image_feature.Gator'
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.4.produce",
        )
        step.add_argument(
            name="outputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.5.produce",
        )
        step.add_hyperparameter(
            name='unfreeze_proportions', argument_type=ArgumentType.VALUE, data=[0.5]
        )
        step.add_hyperparameter(
            name='top_layer_epochs', argument_type=ArgumentType.VALUE, data=top_layer_epochs
        )
        step.add_hyperparameter(
            name='all_layer_epochs', argument_type=ArgumentType.VALUE, data=all_layer_epochs
        )
        step.add_output("produce")
        pipeline_description.add_step(step)
        
        # construct predictions
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_transformation.construct_predictions.Common"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.6.produce",
        )
        step.add_argument(
            name="reference",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.2.produce",
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # Final Output
        pipeline_description.add_output(
            name="output predictions", data_reference="steps.7.produce"
        )

        self.pipeline = pipeline_description
