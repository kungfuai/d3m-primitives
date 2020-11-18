from typing import List

from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

from kf_d3m_primitives.pipeline_base import PipelineBase

class MlpClassifierPipeline(PipelineBase):

    def __init__(
        self, 
        weights_filepath: str = '/scratch_dir/model_weights.pth',
        explain_all_classes: bool = False,
        all_confidences: bool = False, 
        epochs: int = 25
    ):

        pipeline_description = Pipeline()
        pipeline_description.add_input(name="inputs")

        # Denormalize
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_transformation.denormalize.Common"
            )
        )
        step.add_argument(
            name="inputs", 
            argument_type=ArgumentType.CONTAINER, 
            data_reference="inputs.0"
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # DS to DF on input DS
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_transformation.dataset_to_dataframe.Common"
            )
        )
        step.add_argument(
            name="inputs", 
            argument_type=ArgumentType.CONTAINER, 
            data_reference="steps.0.produce"
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # Satellite Image Loader
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_transformation.satellite_image_loader.DistilSatelliteImageLoader"
            )
        )
        step.add_argument(
            name="inputs", 
            argument_type=ArgumentType.CONTAINER, 
            data_reference="steps.1.produce"
        )
        step.add_hyperparameter(
            name="return_result",
            argument_type=ArgumentType.VALUE,
            data="replace"
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # Distil column parser
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_transformation.column_parser.DistilColumnParser"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.2.produce",
        )
        step.add_output("produce")
        step.add_hyperparameter(
            name="parsing_semantics",
            argument_type=ArgumentType.VALUE,
            data=[
                "http://schema.org/Integer",
                "http://schema.org/Float",
                "https://metadata.datadrivendiscovery.org/types/FloatVector",
            ],
        )
        pipeline_description.add_step(step)

        # parse image semantic types
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
        step.add_output("produce")
        step.add_hyperparameter(
            name="semantic_types",
            argument_type=ArgumentType.VALUE,
            data=[
                "http://schema.org/ImageObject",
            ],
        )
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
        step.add_output("produce")
        step.add_hyperparameter(
            name="semantic_types",
            argument_type=ArgumentType.VALUE,
            data=[
                "https://metadata.datadrivendiscovery.org/types/Target",
                "https://metadata.datadrivendiscovery.org/types/TrueTarget"
            ],
        )
        pipeline_description.add_step(step)

        # remote sensing pretrained
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.remote_sensing.remote_sensing_pretrained.RemoteSensingPretrained"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.4.produce",
        )
        step.add_output("produce")
        step.add_hyperparameter(
            name="pool_features",
            argument_type=ArgumentType.VALUE,
            data=False
        )
        pipeline_description.add_step(step)

        # mlp classifier
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.remote_sensing.mlp.MlpClassifier"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.6.produce",
        )
        step.add_argument(
            name="outputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.5.produce",
        )
        step.add_output("produce")
        step.add_hyperparameter(
            name="weights_filepath",
            argument_type=ArgumentType.VALUE,
            data=weights_filepath
        )
        step.add_hyperparameter(
            name="explain_all_classes",
            argument_type=ArgumentType.VALUE,
            data=explain_all_classes
        )
        step.add_hyperparameter(
            name="all_confidences",
            argument_type=ArgumentType.VALUE,
            data=all_confidences
        )
        step.add_hyperparameter(
            name="epochs",
            argument_type=ArgumentType.VALUE,
            data=epochs
        )
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
            data_reference="steps.7.produce",
        )
        step.add_argument(
            name="reference",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.2.produce",
        )
        step.add_output("produce")
        step.add_hyperparameter(
            name="use_columns",
            argument_type=ArgumentType.VALUE,
            data=[0,1]
        )
        pipeline_description.add_step(step)

        pipeline_description.add_output(
            name="output predictions", data_reference="steps.8.produce"
        )

        self.pipeline = pipeline_description