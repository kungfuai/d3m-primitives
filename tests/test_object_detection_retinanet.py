from kf_d3m_primitives.object_detection.retinanet.object_detection_retinanet_pipeline import ObjectDetectionRNPipeline

def _test_serialize(dataset):
    
    pipeline = ObjectDetectionRNPipeline(epochs = 1, n_steps = 1)
    pipeline.write_pipeline()
    pipeline.fit_serialize(dataset)
    pipeline.deserialize_score(dataset)
    pipeline.delete_pipeline()
    pipeline.delete_serialized_pipeline()
    
def test_serialization_dataset_pedestrian():
    _test_serialize('LL1_penn_fudan_pedestrian_MIN_METADATA')

def test_serialization_dataset_terra_panicle():
    _test_serialize('LL1_tidy_terra_panicle_detection_MIN_METADATA')

