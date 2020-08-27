from kf_d3m_primitives.ts_classification.knn.kanine_pipeline import KaninePipeline

def _test_serialize(dataset):
    
    pipeline = KaninePipeline()
    pipeline.write_pipeline()
    pipeline.fit_serialize(dataset)
    pipeline.deserialize_score(dataset)
    pipeline.delete_pipeline()
    pipeline.delete_serialized_pipeline()
    
def test_serialization_dataset_chlorine():
    _test_serialize('66_chlorineConcentration_MIN_METADATA')

