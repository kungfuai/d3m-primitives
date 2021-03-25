from kf_d3m_primitives.data_preprocessing.data_cleaning.data_cleaning_pipeline import (
    DataCleaningPipeline,
)


def _test_fit_score(dataset):

    pipeline = DataCleaningPipeline()
    pipeline.write_pipeline()
    pipeline.fit_score(dataset)
    pipeline.delete_pipeline()


def test_fit_score_dataset_baseball():
    _test_fit_score("185_baseball_MIN_METADATA")
