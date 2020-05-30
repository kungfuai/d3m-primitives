from primitives.image_classification.imagenet_transfer_learning.gator_pipeline import GatorPipeline

def _test_serialize(dataset):
    
    pipeline = GatorPipeline(sample_size = 100, all_layer_epochs = 1, top_layer_epochs = 1)
    pipeline.write_pipeline()
    pipeline.fit_serialize(dataset)
    pipeline.deserialize_score(dataset)
    pipeline.delete_pipeline()
    pipeline.delete_serialized_pipeline()
    
def test_serialization_dataset_cifar():
    _test_serialize('124_174_cifar10_MIN_METADATA')

def test_serialization_dataset_usps():
    _test_serialize('124_188_usps_MIN_METADATA')

def test_serialization_dataset_coil():
    _test_serialize('124_214_coil20_MIN_METADATA')

def test_serialization_dataset_object_categories():
    _test_serialize('uu_101_object_categories_MIN_METADATA')
