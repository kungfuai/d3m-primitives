from primitives.data_preprocessing.data_typing.simon_pipeline import SimonPipeline

def _test_fit_score(dataset):
    
    pipeline = SimonPipeline()
    pipeline.write_pipeline()
    pipeline.fit_serialize(dataset)
    pipeline.deserialize_score(dataset)
    pipeline.delete_pipeline()
    pipeline.delete_serialized_pipeline()
    
def test_fit_score_dataset_baseball():
    _test_fit_score('185_baseball_MIN_METADATA')


