import os.path
import logging
import sys
import random
import struct
import typing

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lzo
from rsp.moco_r50.data import sentinel_augmentation_valid
from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

from .unet import Unet, SegmentationHeadImageLabelEval
from .binary_focal_loss import BinaryFocalLoss

__author__ = 'Distil'
__version__ = '1.0.0'
__contact__ = 'mailto:jeffrey.gleason@kungfu.ai'

Inputs = container.DataFrame
Outputs = container.DataFrame

logger = logging.getLogger(__name__)

class Params(params.Params):
    is_fit: bool
    positive_class: str

class Hyperparams(hyperparams.Hyperparams):
    weights_filepath = hyperparams.Hyperparameter[str](
        default='model_weights.pth',
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="weights of trained model will be saved to this filepath",
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    decompress_data = hyperparams.Hyperparameter[bool](
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="If True, applies LZO decompression algorithm to the data. \
                    Compressed data stores a header consisting of the dtype character and the \
                    data shape as unsigned integers. Given c struct alignment, will occupy \
                    16 bytes (1 + 4 + 4 + 4 + 3 ) padding"
    )
    batch_size = hyperparams.UniformInt(
        lower=1,
        upper=512,
        default=128,
        upper_inclusive=True,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/TuningParameter"],
        description="training and inference batch size",
    )
    val_split = hyperparams.Uniform(
        lower=0.0,
        upper=1.0,
        default=0.1,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/ControlParameter"],
        description="proportion of samples to reserve for validation",
    )
    epochs_frozen = hyperparams.UniformInt(
        lower=0,
        upper=50,
        default=10,
        upper_inclusive=True,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/TuningParameter"],
        description="Number of full training passes with frozen encoder weights",
    )
    epochs_unfrozen = hyperparams.UniformInt(
        lower=0,
        upper=200,
        default=100,
        upper_inclusive=True,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/TuningParameter"],
        description="Number of full training passes with entire model unfrozen",
    )
    patience = hyperparams.UniformInt(
        lower=0,
        upper=50,
        default=10,
        upper_inclusive=True,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/TuningParameter"],
        description="Number of epochs without improvement after which training stops",
    )
    learning_rate = hyperparams.Uniform(
        lower=0.0,
        upper=1.0,
        default=0.001,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/TuningParameter"],
        description="learning rate",
    )

class ImageSegmentationPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """ Primitive that trains a binary image segmentation model using image-level weak supervision 

        Training inputs: 1) Feature dataframe, 2) Label dataframe
        Outputs: D3M dataset with segmentation masks 
    """

    metadata = metadata_base.PrimitiveMetadata({
        'id': "9e01a4a6-67b9-4242-b4a3-ac648937f2bd",
        'version': __version__,
        'name': 'ImageSegmentation',
        'keywords': [
            'remote sensing', 
            'segmentation', 
            'weak supervision', 
            'transfer learning', 
        ],
        'source': {
            'name': __author__,
            'contact': __contact__,
            "uris": [
                "https://github.com/kungfuai/d3m-primitives",
            ],
        },
        "installation": [
            {"type": "PIP", "package": "cython", "version": "0.29.16"}, 
            {
                "type": metadata_base.PrimitiveInstallationType.PIP,
                "package_uri": "git+https://github.com/kungfuai/d3m-primitives.git@{git_commit}#egg=kf-d3m-primitives".format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                ),
            },
            {
                'type': metadata_base.PrimitiveInstallationType.UBUNTU,
                'package': 'zlib1g-dev',
                'version': '1:1.2.11.dfsg-0ubuntu2',
            }, 
            {
                'type': metadata_base.PrimitiveInstallationType.UBUNTU,
                'package': 'liblzo2-dev',
                'version': '2.08-1.2',
            },
            {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'python-lzo',
                'version': '1.12',
            },
            {
            "type": "FILE",
            "key": "moco_weights",
            "file_uri": "http://public.datadrivendiscovery.org/moco_sentinel_v0.pth.tar",
            "file_digest":"fcc8a5a05fa7dbad8fc55584a77fc5d2c407e03a88610267860b45208e152f1f"
            },
        ],
        'python_path': 'd3m.primitives.remote_sensing.convolutional_neural_net.ImageSegmentation',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.DEEP_NEURAL_NETWORK,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.REMOTE_SENSING,
        'can_use_gpus': True
    })

    def __init__(
        self, 
        *, 
        hyperparams: Hyperparams, 
        random_seed: int = 0, 
        volumes: typing.Dict[str, str] = None
    )-> None:

        super().__init__(
            hyperparams=hyperparams, 
            random_seed=random_seed, 
            volumes=volumes
        )

        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._create_model()
        self._compile_model()
        self._is_fit = False

    def get_params(self) -> Params:
        return Params(
            is_fit = self._is_fit,
            positive_class = self._positive_class,
        )

    def set_params(self, *, params: Params) -> None:
        self._is_fit = params['is_fit']
        self._positive_class = params['positive_class']

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """ set primitive's training data.
        
            Arguments:
                inputs {Inputs} -- D3M dataframe containing images
                outputs {Outputs} -- D3M dataframe containing labels
            
        """
        self._load_data(inputs, outputs)
        self._is_fit = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """ Fits binary segmentation model using training data from set_training_data 
            and hyperparameters

            Keyword Arguments:
                timeout {float} -- timeout, considered (default: {None})
                iterations {int} -- iterations, considered (default: {None})

            Returns:
                CallResult[None]
        """

        # freeze encoder
        epochs_elapsed = self._train(self.hyperparams['epochs_frozen'])

        # unfreeze encoder
        for param in self.model.parameters():
            param.requires_grad = True 

        self._train(
            self.hyperparams['epochs_unfrozen'] + epochs_elapsed, 
            initial_epoch=epochs_elapsed
        )
        self._is_fit = True
        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """ produce segmentation masks

            Arguments:
                inputs {Inputs} -- D3M dataframe containing images
            
            Keyword Arguments:
                timeout {float} -- timeout, not considered (default: {None})
                iterations {int} -- iterations, not considered (default: {None})
        """
        
        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        imgs = self._get_imgs(inputs)
        test_loader = self._prepare_loader(imgs, shuffle=False)

        model = Unet(
            encoder_freeze=False,
            device=self.device
        ).to(self.device)

        model.load_state_dict(torch.load(self.hyperparams['weights_filepath']))

        model.segmentation_head = SegmentationHeadImageLabelEval(
            model.segmentation_head
        )

        all_preds = []
        for batch in tqdm(test_loader):
            inputs = batch[0].to(self.device)
            preds = model.predict(inputs).squeeze()
            preds = preds[:, self.pad:-self.pad, self.pad:-self.pad]
            all_preds.append(preds.detach().cpu().numpy())
        all_preds = np.vstack(all_preds)

        preds_df = self._prepare_d3m_df(all_preds)
        return CallResult(preds_df)

    def _load_data(self, inputs, outputs):
        """ load data for training """ 

        imgs = self._get_imgs(inputs)

        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(outputs.values.ravel())
        if label_encoder.classes_.shape[0] > 2:
            raise ValueError('This primitive only supports binary segmentation (2 classes)')
        self._positive_class = label_encoder.inverse_transform([1])[0]
            
        imgs_train, imgs_val, labels_train, labels_val = train_test_split(
            imgs,
            labels,
            test_size=self.hyperparams['val_split'],
            random_state=self.random_seed,
            stratify=labels
        )

        self.train_loader = self._prepare_loader(imgs_train, labels_train)
        self.val_loader = self._prepare_loader(imgs_val, labels_val, shuffle=False)

    def _get_imgs(self, inputs):
        """ get column of imgs from dataset """ 
        
        if len(self.hyperparams['use_columns']) == 0:
            image_cols = inputs.metadata.get_columns_with_semantic_type(
                'http://schema.org/ImageObject'
            )
        else:
            image_cols = self.hyperparams['use_columns']
        
        if len(image_cols) > 1:
            raise ValueError('Primitive only supports fitting model on one image column')

        image_col = image_cols[0]
        imgs = inputs.iloc[:, image_col]

        if self.hyperparams['decompress_data']:
            imgs = [self._decompress(img) for img in imgs]
        
        self.pad = (128 - imgs[0].shape[1]) // 2
        imgs = np.stack([self._prepare_img(img) for img in imgs])

        return imgs
        
    def _decompress(self, img):
        """ decompress image """ 
        compressed_bytes = img.tobytes()
        decompressed_bytes = lzo.decompress(compressed_bytes)
        storage_type, shape_0, shape_1, shape_2 = struct.unpack(
            'cIII', 
            decompressed_bytes[:16]
        )
        img = np.frombuffer(decompressed_bytes[16:], dtype=storage_type)
        img = img.reshape(shape_0, shape_1, shape_2)
        return img

    def _prepare_img(self, img):
        """ normalize and pad image """
        img = img[:12].transpose(1, 2, 0) / 10_000
        img = sentinel_augmentation_valid()(image=img)['image']
        img = torch.nn.functional.pad(
            img, 
            (self.pad,self.pad,self.pad,self.pad,0,0)
        )
        return torch.FloatTensor(img)
    
    def _prepare_loader(self, inputs, outputs=None, shuffle=True):
        """ prepare data loader """

        inputs = torch.FloatTensor(inputs)
        
        if outputs is None:
            dataset = torch.utils.data.TensorDataset(inputs)
        else:
            outputs = torch.FloatTensor(outputs).unsqueeze(1) 
            dataset = torch.utils.data.TensorDataset(inputs, outputs)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hyperparams['batch_size'],
            shuffle=shuffle
        )
        return data_loader

    def _create_model(self):
        """ construct Unet segmentation model """ 
        
        self.model = Unet(
            encoder_weights=self.volumes['moco_weights'],
            device=self.device
        ).to(self.device)

        if os.path.isfile(self.hyperparams['weights_filepath']):
            self.model.load_state_dict(
                torch.load(self.hyperparams['weights_filepath'])
            )

    def _compile_model(self):
        """ construct loss function, optimizer, and learning rate scheduler """
        
        self.loss = BinaryFocalLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hyperparams['learning_rate']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            verbose=True
        )

    def _train(self, epochs, initial_epoch=0):
        """ train model """

        if epochs == 0:
            return 0

        stopping_ct = 0
        stopping_loss = None

        for epoch in range(initial_epoch, epochs):

            train_losses = []
            train_accs = []
            val_losses = []
            val_accs = []

            self.model.train()
            train_bs = []
            for batch in tqdm(self.train_loader):

                self.optimizer.zero_grad()
                train_loss, train_acc = self._loss(batch)
                train_loss.backward()
                self.optimizer.step()

                train_losses.append(train_loss.item())
                train_accs.append(train_acc)
                train_bs.append(batch[1].shape[0])

            self.model.eval()
            val_bs = []
            with torch.no_grad():
                for batch in tqdm(self.val_loader):
                    loss, acc = self._loss(batch)
                    val_losses.append(loss.item())
                    val_accs.append(acc)
                    val_bs.append(batch[1].shape[0])
            
            train_loss = np.average(train_losses, weights=train_bs)
            train_acc = np.average(train_accs, weights=train_bs)
            val_loss = np.average(val_losses, weights=val_bs)
            val_acc = np.average(val_accs, weights=val_bs)

            self.scheduler.step(val_loss)               

            # self.logger
            print(
                f'Epoch {epoch}, ' +
                f'Train Loss: {round(train_loss, 3)}',
                f'Train Acc: {round(train_acc, 3)}',
                f'Val Loss: {round(val_loss, 3)}',
                f'Val Acc: {round(val_acc, 3)}'
            )

            if stopping_loss is None:
                stopping_loss = val_loss
                torch.save(self.model.state_dict(), self.hyperparams['weights_filepath'])
            elif stopping_loss <= val_loss:
                stopping_ct += 1
            else:
                stopping_loss = val_loss
                stopping_ct = 0
                torch.save(self.model.state_dict(), self.hyperparams['weights_filepath'])
            
            if stopping_ct == self.hyperparams['patience']:
                ## self.logger
                print(f"Stopping training early - no improvement for {self.hyperparams['patience']} epochs")
                break

        return epoch + 1

    def _loss(self, batch):
        """ compute loss and accuracy on batch """
        inputs = batch[0].to(self.device)
        labels = batch[1].to(self.device)
        outputs = self.model(inputs)
        loss = self.loss(outputs, labels)

        preds = torch.round(outputs)
        acc = (preds == labels)

        loss = loss.mean()
        acc = acc.sum().item() / labels.shape[0]

        return loss, acc

    def _prepare_d3m_df(self, all_preds):
        """ prepare d3m dataframe with appropriate metadata """ 

        all_preds = [preds.tolist() for preds in all_preds]
        preds_df = pd.DataFrame({f'{self._positive_class}_mask': all_preds})
        preds_df = d3m_DataFrame(preds_df, generate_metadata=False)
        preds_df.metadata = preds_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0),
            "https://metadata.datadrivendiscovery.org/types/FloatVector"
        )
        return preds_df