from kf_d3m_primitives.dimensionality_reduction.tsne.tsne_pipeline import TsnePipeline


def _test_fit_score(dataset):

    pipeline = TsnePipeline()
    pipeline.write_pipeline()
    pipeline.fit_score(dataset)
    pipeline.delete_pipeline()


def test_fit_score_dataset_semi_supervised():
    _test_fit_score("SEMI_1044_eye_movements_MIN_METADATA")
