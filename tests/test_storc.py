from primitives.clustering.k_means.storc_pipeline import StorcPipeline

def _test_fit_produce(dataset):
    
    pipeline = StorcPipeline()
    pipeline.write_pipeline()
    pipeline.fit_produce(dataset)
    pipeline.delete_pipeline()

def test_fit_produce_dataset_chlorine():
    _test_fit_produce('66_chlorineConcentration_MIN_METADATA')

