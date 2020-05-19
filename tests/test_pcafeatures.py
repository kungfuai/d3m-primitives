from primitives.feature_selection.pca_features.pca_features_pipeline import PcaFeaturesPipeline

def _test_serialize(dataset):
    
    pipeline = PcaFeaturesPipeline()
    pipeline.write_pipeline()
    pipeline.fit_serialize(dataset)
    pipeline.deserialize_score(dataset)
    pipeline.delete_pipeline()
    pipeline.delete_serialized_pipeline()
    
def test_serialization_dataset_baseball():
    _test_serialize('185_baseball_MIN_METADATA')


