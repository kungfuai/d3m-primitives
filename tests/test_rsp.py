import unittest
import os

import pandas as pd
import numpy as np
from rsp.data import load_patch

from primitives.remote_sensing.featurizer.remote_sensing_pretrained import RemoteSensingPretrainedPrimitive

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
    
    def _load_frame(self):
        img_paths = [
            os.path.join(self.dataset_path, filename) 
            for filename in os.listdir(self.dataset_path)
        ]
        imgs = [
            load_patch(img_path).astype(np.float32) 
            for img_path in img_paths
        ]
        return pd.DataFrame({'image_col': imgs})

    def _test_rsp(self, model, weight_path, output_dim):

        test_frame = self._load_frame()
        rs_hp = RemoteSensingPretrainedPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        rsp = RemoteSensingPretrainedPrimitive(
            hyperparams=rs_hp(
                rs_hp.defaults(),
                inference_model = model,
                use_columns = [0],
            ),
            volumes = {'{}_weights'.format(model): weight_path},
        )
        feature_df = rsp.produce(inputs=test_frame).value
        self.assertEqual(feature_df.shape[0], test_frame.shape[0])
        self.assertEqual(feature_df.shape[1], output_dim)

if __name__ == '__main__':
    unittest.main()
