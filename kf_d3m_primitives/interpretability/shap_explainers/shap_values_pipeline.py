from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

from kf_d3m_primitives.pipeline_base import PipelineBase


class ShapPipeline(PipelineBase):
    def __init__(self):

        pipeline_description = Pipeline()
        pipeline_description.add_input(name="inputs")

        # dataset_to_dataframe
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

        # column parser -> labeled semantic types to data types
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

        # imputer -> imputes null values based on mean of column
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_cleaning.imputer.SKlearn"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.2.produce",
        )
        step.add_hyperparameter(
            name="return_result", argument_type=ArgumentType.VALUE, data="replace"
        )
        step.add_hyperparameter(
            name="use_semantic_types", argument_type=ArgumentType.VALUE, data=True
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # extract feature columns
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
            data=("https://metadata.datadrivendiscovery.org/types/Attribute",),
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # extract target columns
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
            data=(
                "https://metadata.datadrivendiscovery.org/types/Target",
                "https://metadata.datadrivendiscovery.org/types/TrueTarget",
            ),
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # Text encoder
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_transformation.encoder.DistilTextEncoder"
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
            name="metric",
            argument_type=ArgumentType.VALUE,
            data="accuracy",
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # Random forest shap values
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.learner.random_forest.DistilEnsembleForest"
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
        step.add_output("produce_shap_values")
        pipeline_description.add_step(step)

        # Final Output
        pipeline_description.add_output(
            name="output", data_reference="steps.7.produce_shap_values"
        )

        self.pipeline = pipeline_description