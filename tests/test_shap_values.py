from primitives.interpretability.shap_explainers.shap_values_pipeline import ShapPipeline

def _test_fit_produce(dataset):
    
    pipeline = ShapPipeline()
    pipeline.write_pipeline()
    pipeline.fit_produce(dataset)
    pipeline.delete_pipeline()

def test_fit_produce_dataset_baseball():
    _test_fit_produce('185_baseball_MIN_METADATA')

def test_fit_produce_dataset_acled():
    _test_fit_produce('LL0_acled_reduced_MIN_METADATA')
