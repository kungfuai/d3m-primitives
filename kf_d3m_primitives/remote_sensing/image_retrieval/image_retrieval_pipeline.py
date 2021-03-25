from typing import List
import json
import os
import time
import subprocess

import numpy as np
import pandas as pd
from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.container import DataFrame as d3m_DataFrame

from kf_d3m_primitives.pipeline_base import PipelineBase


class ImageRetrievalPipeline(PipelineBase):
    def __init__(
        self,
        annotations: List[int] = None,
        gem_p: int = 1,
        dataset: str = "LL1_bigearth_landuse_detection",
    ):

        pipeline_description = Pipeline()
        pipeline_description.add_input(name="inputs")
        if annotations is None:
            pipeline_description.add_input(name="annotations")

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
        # TODO test without index
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
                "https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey",
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
        pipeline_description.add_step(step)

        # DS to DF on annotations DS
        if annotations is None:
            step = PrimitiveStep(
                primitive=index.get_primitive(
                    "d3m.primitives.data_transformation.dataset_to_dataframe.Common"
                )
            )
            step.add_argument(
                name="inputs",
                argument_type=ArgumentType.CONTAINER,
                data_reference="inputs.1",
            )
            step.add_output("produce")
            pipeline_description.add_step(step)

        # image retrieval primitive
        step = PrimitiveStep(
            primitive=index.get_primitive(
                "d3m.primitives.similarity_modeling.iterative_labeling.ImageRetrieval"
            )
        )
        step.add_argument(
            name="inputs",
            argument_type=ArgumentType.CONTAINER,
            data_reference="steps.5.produce",
        )
        if annotations is not None:
            step.add_argument(
                name="outputs",
                argument_type=ArgumentType.VALUE,
                data=annotations,
            )
        else:
            step.add_argument(
                name="outputs",
                argument_type=ArgumentType.CONTAINER,
                data_reference="steps.6.produce",
            )
        step.add_output("produce")
        step.add_hyperparameter(
            name="gem_p", argument_type=ArgumentType.VALUE, data=gem_p
        )
        pipeline_description.add_step(step)

        if annotations is not None:
            pipeline_description.add_output(
                name="output ranking", data_reference="steps.6.produce"
            )
        else:
            pipeline_description.add_output(
                name="output ranking", data_reference="steps.7.produce"
            )

        self.pipeline = pipeline_description
        self.dataset = dataset
        self.annotations = annotations

    def make_annotations_dataset(self, n_rows, round_num=0, num_bands=12):

        if self.annotations is None:
            annotationsDoc = {
                "dataResources": [
                    {
                        "resID": "annotationsData",
                        "resPath": "/scratch_dir/annotationsData.csv",
                        "resType": "table",
                        "resFormat": {"text/csv": ["csv"]},
                        "columns": [
                            {
                                "colIndex": 0,
                                "colName": "annotations",
                                "colType": "integer",
                                "role": ["attribute"],
                            }
                        ],
                    }
                ]
            }
            with open("/scratch_dir/annotationsDoc.json", "w") as json_file:
                json.dump(annotationsDoc, json_file)

        if round_num == 0:
            annotations = np.zeros(n_rows) - 1
            annotations[0] = 1
            annotations = pd.DataFrame(
                {"d3mIndex": np.arange(n_rows), "annotations": annotations.astype(int)}
            )
        else:
            annotations = pd.read_csv("/scratch_dir/annotationsData.csv")
            ranking = pd.read_csv("/scratch_dir/rankings.csv")
            test_index = pd.read_csv(
                f"/datasets/seed_datasets_current/{self.dataset}/TEST/dataset_TEST/tables/learningData.csv"
            )["d3mIndex"].values

            top_idx = np.where(test_index == ranking.iloc[0, 0])[0][0] // num_bands
            human_annotation = np.random.randint(2)
            annotations.iloc[top_idx, 1] = human_annotation

        annotations.to_csv("/scratch_dir/annotationsData.csv", index=False)
        return d3m_DataFrame(annotations)

    def delete_annotations_dataset(self):

        if self.annotations is None:
            subprocess.run(["rm", "/scratch_dir/annotationsDoc.json"], check=True)

        subprocess.run(["rm", "/scratch_dir/annotationsData.csv"], check=True)
        subprocess.run(["rm", "/scratch_dir/rankings.csv"], check=True)

    def fit_produce(self, output_yml_dir=".", submission=False):

        if not os.path.isfile(self.outfile_string):
            raise ValueError("Must call 'write_pipeline()' method first")

        proc_cmd = [
            "python3",
            "-m",
            "d3m",
            "runtime",
            "-d",
            "/datasets",
            "-v",
            "/static_volumes",
            "-s",
            "/scratch_dir",
            "fit-produce",
            "-p",
            self.outfile_string,
            "-i",
            f"/datasets/seed_datasets_current/{self.dataset}/TEST/dataset_TEST/datasetDoc.json",
            "-r",
            f"/datasets/seed_datasets_current/{self.dataset}/{self.dataset}_problem/problemDoc.json",
            "-t",
            f"/datasets/seed_datasets_current/{self.dataset}/TEST/dataset_TEST/datasetDoc.json",
            "-o",
            f"/scratch_dir/rankings.csv",
        ]

        if self.annotations is None:
            proc_cmd += [
                "-i",
                "/scratch_dir/annotationsData.csv",
                "-t",
                "/scratch_dir/annotationsData.csv",
            ]

        if submission:
            proc_cmd += ["-O", f"{output_yml_dir}/{self.dataset}.yml"]

        st = time.time()
        subprocess.run(proc_cmd, check=True)
        print(f"Fitting and producing pipeline took {(time.time() - st) / 60} mins")
