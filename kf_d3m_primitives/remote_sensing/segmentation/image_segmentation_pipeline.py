from typing import List
import os
import subprocess
import time

from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

from kf_d3m_primitives.pipeline_base import PipelineBase


class ImageSegmentationPipeline(PipelineBase):
    def __init__(
        self,
        binary_labels,
        weights_filepath: str = "scratch_dir/model_weights.pth",
        epochs_frozen: int = 20,
        epochs_unfrozen: int = 100,
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

        # Satellite Image Loader
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.data_transformation.satellite_image_loader.DistilSatelliteImageLoader"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.1.produce",
        )
        step.add_hyperparameter(
            name="return_result", argument_type=ArgumentType.VALUE, data="replace"
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

        # image segmentation
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.remote_sensing.convolutional_neural_net.ImageSegmentation"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.4.produce",
        )
        step.add_argument(
            name="outputs",
            argument_type=ArgumentType.VALUE,
            data=binary_labels,
        )
        step.add_output("produce")
        step.add_hyperparameter(
            name="weights_filepath",
            argument_type=ArgumentType.VALUE,
            data=weights_filepath,
        )
        step.add_hyperparameter(
            name="epochs_frozen", argument_type=ArgumentType.VALUE, data=epochs_frozen
        )
        step.add_hyperparameter(
            name="epochs_unfrozen",
            argument_type=ArgumentType.VALUE,
            data=epochs_unfrozen,
        )
        pipeline_description.add_step(step)

        pipeline_description.add_output(
            name="output predictions", data_reference="steps.5.produce"
        )

        self.pipeline = pipeline_description

    def fit_produce(self, dataset, output_yml_dir=".", submission=False):

        if not os.path.isfile(self.outfile_string):
            raise ValueError("Must call 'write_pipeline()' method first")

        proc_cmd = [
            "python3",
            "-m",
            "d3m",
            "runtime",
            "-D",
            "/datasets",
            "-v",
            "/static_volumes",
            "--scratch",
            "/scratch_dir",
            "fit-produce",
            "-p",
            self.outfile_string,
            "-i",
            f"/datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json",
            "-r",
            f"/datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json",
            "-t",
            f"/datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json",
        ]
        if submission:
            proc_cmd += ["-O", f"{output_yml_dir}/{dataset}.yml"]

        st = time.time()
        subprocess.run(proc_cmd, check=True)
        print(f"Fitting and producing pipeline took {(time.time() - st) / 60} mins")
