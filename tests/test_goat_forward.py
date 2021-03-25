from kf_d3m_primitives.data_preprocessing.geocoding_forward.goat_forward_pipeline import (
    GoatForwardPipeline,
)


def _test_fit_score(dataset):

    pipeline = GoatForwardPipeline()
    pipeline.write_pipeline()
    pipeline.fit_score(dataset)
    pipeline.delete_pipeline()


def test_fit_score_dataset_acled():
    _test_fit_score("LL0_acled_reduced_MIN_METADATA")
