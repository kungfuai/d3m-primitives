from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

from kf_d3m_primitives.pipeline_base import PipelineBase


class GoatReversePipeline(PipelineBase):
    def __init__(self):

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

        # Simon
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_cleaning.column_type_profiler.Simon"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.0.produce",
        )
        step.add_hyperparameter(
            name="statistical_classification",
            argument_type=ArgumentType.VALUE,
            data=False,
        )
        step.add_hyperparameter(
            name="p_threshold", argument_type=ArgumentType.VALUE, data=0.1
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # column parser
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
        pipeline_description.add_step(step)

        # Goat reverse
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_cleaning.geocoding.Goat_reverse"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.2.produce",
        )
        step.add_hyperparameter(
            name="cache_size", argument_type=ArgumentType.VALUE, data=2000
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # XG Boost
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.classification.xgboost_gbtree.Common"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.3.produce",
        )
        step.add_argument(
            name="outputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.3.produce",
        )
        step.add_output("produce")
        step.add_hyperparameter(
            name="return_result", argument_type=ArgumentType.VALUE, data="replace"
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
            data_reference="steps.4.produce",
        )
        step.add_argument(
            name="reference",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.1.produce",
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # Final Output
        pipeline_description.add_output(
            name="output predictions", data_reference="steps.5.produce"
        )

        self.pipeline = pipeline_description
