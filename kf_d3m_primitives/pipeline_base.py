import subprocess
import os
import time
import json


class PipelineBase:
    def __init__(self):
        raise NotImplementedError

    def write_pipeline(self, output_dir="."):
        json_pipeline = self.pipeline.to_json()
        self.outfile_string = f'{output_dir}/{json.loads(json_pipeline)["id"]}.json'
        with open(self.outfile_string, "w") as outfile:
            outfile.write(json_pipeline)

    def fit_serialize(self, dataset):

        if not os.path.isfile(self.outfile_string):
            raise ValueError("Must call 'write_pipeline()' method first")

        self.serialized_file = f"test_pipeline_{dataset}.d3m"
        subprocess.run(
            [
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
                "fit",
                "-p",
                self.outfile_string,
                "-s",
                f"test_pipeline_{dataset}.d3m",
                "-i",
                f"/datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json",
                "-r",
                f"/datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json",
            ],
            check=True,
        )

    def deserialize_score(self, dataset):

        if not os.path.isfile(self.serialized_file):
            raise ValueError(f"The file 'test_pipeline_{dataset}.d3m' does not exist.")

        subprocess.run(
            [
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
                "score",
                "-f",
                f"test_pipeline_{dataset}.d3m",
                "-t",
                f"/datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json",
                "-a",
                f"/datasets/seed_datasets_current/{dataset}/SCORE/dataset_SCORE/datasetDoc.json",
            ],
            check=True,
        )

    def fit_produce(self, dataset, output_yml_dir=".", submission=False):

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
            f"/datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json",
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

    def fit_produce_all(self, dataset):

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
            f"/datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json",
            "-r",
            f"/datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json",
            "-t",
            f"/datasets/seed_datasets_current/{dataset}/{dataset}_dataset/datasetDoc.json",
        ]

        st = time.time()
        subprocess.run(proc_cmd, check=True)
        print(f"Fitting and producing pipeline took {(time.time() - st) / 60} mins")

    def fit_score(
        self,
        dataset,
        output_yml_dir=".",
        output_score_dir=".",
        submission=False,
        suffix=None,
    ):

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
            "fit-score",
            "-p",
            self.outfile_string,
            "-i",
            f"/datasets/seed_datasets_current/{dataset}/TRAIN/dataset_TRAIN/datasetDoc.json",
            "-r",
            f"/datasets/seed_datasets_current/{dataset}/{dataset}_problem/problemDoc.json",
            "-t",
            f"/datasets/seed_datasets_current/{dataset}/TEST/dataset_TEST/datasetDoc.json",
            "-a",
            f"/datasets/seed_datasets_current/{dataset}/SCORE/dataset_SCORE/datasetDoc.json",
        ]
        if submission:
            if suffix:
                self.scorefile = f"{output_score_dir}/{dataset}_{suffix}_scores.csv"
                proc_cmd += [
                    "-O",
                    f"{output_yml_dir}/{dataset}_{suffix}.yml",
                    "-c",
                    self.scorefile,
                ]
            else:
                self.scorefile = f"{output_score_dir}/{dataset}_scores.csv"
                proc_cmd += [
                    "-O",
                    f"{output_yml_dir}/{dataset}.yml",
                    "-c",
                    self.scorefile,
                ]

        st = time.time()
        subprocess.run(proc_cmd, check=True)
        print(f"Fitting and scoring pipeline took {(time.time() - st) / 60} mins")

    def delete_pipeline(self):
        subprocess.run(["rm", self.outfile_string], check=True)

    def delete_serialized_pipeline(self):
        subprocess.run(["rm", self.serialized_file], check=True)
