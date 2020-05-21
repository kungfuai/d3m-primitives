from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

from primitives.pipeline_base import PipelineBase

class TsnePipeline(PipelineBase):

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
            name="inputs", argument_type=ArgumentType.CONTAINER, data_reference="inputs.0"
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # Simple Profiler Column Role Annotation
        step = PrimitiveStep(
            primitive=index.get_primitive("d3m.primitives.schema_discovery.profiler.Common")
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

        # imputer
        step = PrimitiveStep(
            primitive=index.get_primitive("d3m.primitives.data_cleaning.imputer.SKlearn")
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.2.produce",
        )
        step.add_output("produce")
        step.add_hyperparameter(
            name="return_result", argument_type=ArgumentType.VALUE, data="replace"
        )
        step.add_hyperparameter(
            name="use_semantic_types", argument_type=ArgumentType.VALUE, data=True
        )
        pipeline_description.add_step(step)

        # TSNE
        step = PrimitiveStep(
            primitive=index.get_primitive(
                'd3m.primitives.dimensionality_reduction.t_distributed_stochastic_neighbor_embedding.Tsne'
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.3.produce",
        )
        step.add_hyperparameter(
            name='n_components', argument_type=ArgumentType.VALUE, data=3
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # R Forest
        step = PrimitiveStep(
            primitive=index.get_primitive(
                'd3m.primitives.classification.random_forest.SKlearn'
            )
        )
        step.add_argument(
            name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce'
        )
        step.add_argument(
            name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce'
        )
        step.add_output('produce')
        step.add_hyperparameter(
            name='add_index_columns', argument_type=ArgumentType.VALUE,data=True
        )
        step.add_hyperparameter(
            name='use_semantic_types', argument_type=ArgumentType.VALUE,data=True
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
            data_reference="steps.5.produce",
        )
        step.add_argument(
            name="reference",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.0.produce",
        )
        step.add_output("produce")
        pipeline_description.add_step(step)

        # Final Output
        pipeline_description.add_output(
            name="output predictions", data_reference="steps.6.produce"
        )

        self.pipeline = pipeline_description