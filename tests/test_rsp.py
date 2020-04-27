import unittest
import os
import pandas as pd
from primitives.remote_sensing.featurizer.pretrained_featurizer import RemoteSensingTransferPrimitive
#from d3m.primitives.remote_sensing.remote_sensing_pretrained import RemoteSensingTransfer

class RemoteSensingTransferPrimitiveTestCase(unittest.TestCase):

    dataset_path = '/test_data/BigEarthNet-trimmed'
    
    ## produced by downloading weights with https://github.com/cfld/rs_pretrained/blob/master/install.sh (on Ubuntu)
    ## then running sha256sum on downloaded files to prevent metadata load d3m errors
    amdim_path = '/static_volumes/8946fea864c29ed785e00a9cbaa9a50295eb5a334b014f27ba20927104b07f46'
    moco_path = '/static_volumes/fcc8a5a05fa7dbad8fc55584a77fc5d2c407e03a88610267860b45208e152f1f'

    def test_amdim(self):
        self._test_rsp('amdim', self.amdim_path, 1024)

    def test_moco(self):
        self._test_rsp('moco', self.moco_path, 2048)
    
    def _test_rsp(self, model, weight_path, output_dim):

        test_frame = pd.DataFrame(os.listdir(self.dataset_path))
        rs_hp = RemoteSensingTransferPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        rsp = RemoteSensingTransferPrimitive(
            hyperparams=rs_hp(
                rs_hp.defaults(),
                inference_model = model,
                use_columns = [0],
                base_path = self.dataset_path
            ),
            volumes = {'{}_weights'.format(model): weight_path},
        )
        feature_df = rsp.produce(inputs=test_frame).value
        self.assertEqual(feature_df.shape[0], test_frame.shape[0])
        self.assertEqual(feature_df.shape[1], output_dim)

if __name__ == '__main__':
    unittest.main()
