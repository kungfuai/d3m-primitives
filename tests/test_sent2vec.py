from kf_d3m_primitives.natural_language_processing.sent2vec.sent2vec_pipeline import Sent2VecPipeline

def _test_fit_score(dataset):
    
    pipeline = Sent2VecPipeline()
    pipeline.write_pipeline()
    pipeline.fit_score(dataset)
    pipeline.delete_pipeline()

def test_fit_score_dataset_apple_products():
    _test_fit_score('LL1_TXT_CLS_apple_products_sentiment_MIN_METADATA')


