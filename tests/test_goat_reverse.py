from primitives.data_preprocessing.geocoding_reverse.goat_reverse_pipeline import GoatReversePipeline

def _test_fit_score(dataset):
    
    pipeline = GoatReversePipeline()
    pipeline.write_pipeline()
    pipeline.fit_score(dataset)
    pipeline.delete_pipeline()

def test_fit_score_dataset_acled():
    _test_fit_score('LL0_acled_reduced_MIN_METADATA')


