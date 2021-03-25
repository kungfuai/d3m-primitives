from kf_d3m_primitives.feature_selection.rf_features.rf_features_pipeline import (
    RfFeaturesPipeline,
)


def _test_serialize(dataset):

    pipeline = RfFeaturesPipeline()
    pipeline.write_pipeline()
    pipeline.fit_serialize(dataset)
    pipeline.deserialize_score(dataset)
    pipeline.delete_pipeline()
    pipeline.delete_serialized_pipeline()


def test_serialization_dataset_baseball():
    _test_serialize("185_baseball_MIN_METADATA")
