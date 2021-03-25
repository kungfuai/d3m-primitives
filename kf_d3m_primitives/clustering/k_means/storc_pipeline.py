from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

from kf_d3m_primitives.pipeline_base import PipelineBase


class StorcPipeline(PipelineBase):
    def __init__(self):

        pipeline_description = Pipeline()
        pipeline_description.add_input(name="inputs")

        # Ts formatter
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_transformation.time_series_formatter.DistilTimeSeriesFormatter"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="inputs.0",
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # DS to DF on formatted ts DS
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

        # Grouping Field Compose
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_transformation.grouping_field_compose.Common"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.1.produce",
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # Storc primitive -> KMeans
        step = PrimitiveStep(
            primitive=index.get_primitive("d3m.primitives.clustering.k_means.Sloth")
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.2.produce",
        )
        step.add_hyperparameter(
            name="nclusters", argument_type=ArgumentType.VALUE, data=3
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # Final Output
        pipeline_description.add_output(
            name="output predictions", data_reference="steps.3.produce"
        )

        self.pipeline = pipeline_description
