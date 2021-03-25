from kf_d3m_primitives.ts_classification.lstm_fcn.lstm_fcn_pipeline import (
    LstmFcnPipeline,
)


def _test_serialize(dataset):

    pipeline = LstmFcnPipeline(epochs=1)
    pipeline.write_pipeline()
    pipeline.fit_serialize(dataset)
    pipeline.deserialize_score(dataset)
    pipeline.delete_pipeline()
    pipeline.delete_serialized_pipeline()


def test_serialization_dataset_chlorine():
    _test_serialize("66_chlorineConcentration_MIN_METADATA")
