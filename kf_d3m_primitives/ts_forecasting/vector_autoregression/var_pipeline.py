from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

from kf_d3m_primitives.pipeline_base import PipelineBase


class VarPipeline(PipelineBase):
    def __init__(
        self,
        group_compose: bool = False,
        confidence_intervals: bool = False,
        produce_weights: bool = False,
        interpret_pooling: str = "avg",
        interpret_value: str = "lag_order",
    ):

        pipeline_description = Pipeline()
        pipeline_description.add_input(name="inputs")

        # DS to DF on input DS
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_transformation.dataset_to_dataframe.Common"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="inputs.0",
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # Simple Profiler Column Role Annotation
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.schema_discovery.profiler.Common"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.0.produce",
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
            data_reference="steps.1.produce",
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
                "http://schema.org/DateTime",
            ],
        )
        pipeline_description.add_step(step)

        # group compose
        if group_compose:
            step = PrimitiveStep(
                primitive=index.get_primitive(
                    "d3m.primitives.data_transformation.grouping_field_compose.Common"
                )
            )
            step.add_argument(
                name="inputs",
                argument_type=ArgumentType.CONTAINER,
                data_reference="steps.2.produce",
            )
            step.add_output("produce")
            pipeline_description.add_step(step)

        # parse attribute and index semantic types
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common"
            )
        )
        data_ref = "steps.3.produce" if group_compose else "steps.2.produce"
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference=data_ref,
        )
        step.add_hyperparameter(
            name="semantic_types",
            argument_type=ArgumentType.VALUE,
            data=[
                "https://metadata.datadrivendiscovery.org/types/Attribute",
                "https://metadata.datadrivendiscovery.org/types/GroupingKey",
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
        data_ref = "steps.3.produce" if group_compose else "steps.2.produce"
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference=data_ref,
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

        # forecasting primitive
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.time_series_forecasting.vector_autoregression.VAR"
            )
        )
        data_ref = "steps.4.produce" if group_compose else "steps.3.produce"
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference=data_ref,
        )
        data_ref = "steps.5.produce" if group_compose else "steps.4.produce"
        step.add_argument(
            name="outputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference=data_ref,
        )
        step.add_hyperparameter(
            name="interpret_value",
            argument_type=ArgumentType.VALUE,
            data=interpret_value,
        )
        step.add_hyperparameter(
            name="interpret_pooling",
            argument_type=ArgumentType.VALUE,
            data=interpret_pooling,
        )
        if confidence_intervals:
            step.add_output("produce_confidence_intervals")
            pipeline_description.add_step(step)

            data_ref = (
                "steps.6.produce_confidence_intervals"
                if group_compose
                else "steps.5.produce_confidence_intervals"
            )
            pipeline_description.add_output(name="output", data_reference=data_ref)
        elif produce_weights:
            step.add_output("produce_weights")
            pipeline_description.add_step(step)

            data_ref = (
                "steps.6.produce_weights"
                if group_compose
                else "steps.5.produce_weights"
            )
            pipeline_description.add_output(name="output", data_reference=data_ref)
        else:
            step.add_output("produce")
            pipeline_description.add_step(step)

            # construct predictions
            step = PrimitiveStep(
                primitive=index.get_primitive(
                    "d3m.primitives.data_transformation.construct_predictions.Common"
                )
            )
            data_ref = "steps.6.produce" if group_compose else "steps.5.produce"
            step.add_argument(
                name="inputs",
                argument_type=ArgumentType.CONTAINER,
                data_reference=data_ref,
            )
            step.add_argument(
                name="reference",
                argument_type=ArgumentType.CONTAINER,
                data_reference="steps.1.produce",
            )
            step.add_output("produce")
            pipeline_description.add_step(step)

            data_ref = "steps.7.produce" if group_compose else "steps.6.produce"
            pipeline_description.add_output(
                name="output predictions", data_reference=data_ref
            )

        self.pipeline = pipeline_description
