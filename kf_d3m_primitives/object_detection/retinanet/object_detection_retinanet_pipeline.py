from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

from kf_d3m_primitives.pipeline_base import PipelineBase


class ObjectDetectionRNPipeline(PipelineBase):
    def __init__(self, epochs: int = 10, n_steps: int = 20):

        pipeline_description = Pipeline()
        pipeline_description.add_input(name="inputs")

        # Denormalize primitive
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_transformation.denormalize.Common"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="inputs.0",
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
            data_reference="steps.0.produce",
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # RetinaNet primitive
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.object_detection.retina_net.ObjectDetectionRN"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.1.produce",
        )
        step.add_argument(
            name="outputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.1.produce",
        )
        step.add_hyperparameter(
            name="n_epochs", argument_type=ArgumentType.VALUE, data=epochs
        )
        step.add_hyperparameter(
            name="n_steps", argument_type=ArgumentType.VALUE, data=n_steps
        )
        step.add_hyperparameter(
            name="weights_path", argument_type=ArgumentType.VALUE, data="/scratch_dir/"
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # Final Output
        pipeline_description.add_output(
            name="output predictions", data_reference="steps.2.produce"
        )

        self.pipeline = pipeline_description
