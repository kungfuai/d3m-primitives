from primitives.data_preprocessing.text_summarization.duke_pipeline import DukePipeline

def _test_fit_produce(dataset):
    
    pipeline = DukePipeline()
    pipeline.write_pipeline()
    pipeline.fit_produce(dataset)
    pipeline.delete_pipeline()

def test_fit_score_dataset_baseball():
    _test_fit_produce('185_baseball_MIN_METADATA')


