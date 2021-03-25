from kf_d3m_primitives.clustering.spectral_clustering.spectral_clustering_pipeline import (
    SpectralClusteringPipeline,
)


def _test_fit_score(dataset):

    pipeline = SpectralClusteringPipeline()
    pipeline.write_pipeline()
    pipeline.fit_score(dataset)
    pipeline.delete_pipeline()


def test_fit_score_dataset_semi_supervised():
    _test_fit_score("SEMI_1044_eye_movements_MIN_METADATA")
