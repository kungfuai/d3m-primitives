"""
Utility to get generate all submission pipelines for all primitives. 

This script assumes that `generate_annotations.py` has already been run.
"""

import os
import subprocess
import shutil

import fire

from kf_d3m_primitives.data_preprocessing.data_cleaning.data_cleaning_pipeline import DataCleaningPipeline
from kf_d3m_primitives.data_preprocessing.text_summarization.duke_pipeline import DukePipeline
from kf_d3m_primitives.data_preprocessing.geocoding_forward.goat_forward_pipeline import GoatForwardPipeline
from kf_d3m_primitives.data_preprocessing.geocoding_reverse.goat_reverse_pipeline import GoatReversePipeline
from kf_d3m_primitives.data_preprocessing.data_typing.simon_pipeline import SimonPipeline
from kf_d3m_primitives.clustering.spectral_clustering.spectral_clustering_pipeline import SpectralClusteringPipeline
from kf_d3m_primitives.clustering.k_means.storc_pipeline import StorcPipeline
from kf_d3m_primitives.clustering.hdbscan.hdbscan_pipeline import HdbscanPipeline
from kf_d3m_primitives.dimensionality_reduction.tsne.tsne_pipeline import TsnePipeline
from kf_d3m_primitives.feature_selection.pca_features.pca_features_pipeline import PcaFeaturesPipeline
from kf_d3m_primitives.feature_selection.rf_features.rf_features_pipeline import RfFeaturesPipeline
from kf_d3m_primitives.natural_language_processing.sent2vec.sent2vec_pipeline import Sent2VecPipeline
from kf_d3m_primitives.object_detection.retinanet.object_detection_retinanet_pipeline import ObjectDetectionRNPipeline
from kf_d3m_primitives.image_classification.imagenet_transfer_learning.gator_pipeline import GatorPipeline
from kf_d3m_primitives.ts_classification.knn.kanine_pipeline import KaninePipeline
from kf_d3m_primitives.ts_classification.lstm_fcn.lstm_fcn_pipeline import LstmFcnPipeline
from kf_d3m_primitives.ts_forecasting.vector_autoregression.var_pipeline import VarPipeline
from kf_d3m_primitives.ts_forecasting.deep_ar.deepar_pipeline import DeepARPipeline
from kf_d3m_primitives.ts_forecasting.nbeats.nbeats_pipeline import NBEATSPipeline
from kf_d3m_primitives.remote_sensing.classifier.mlp_classifier_pipeline import MlpClassifierPipeline

def generate_pipelines(gpu = False):

    gpu_prims = [
        "d3m.primitives.classification.inceptionV3_image_feature.Gator",
        "d3m.primitives.object_detection.retina_net.ObjectDetectionRN",
        "d3m.primitives.time_series_classification.convolutional_neural_net.LSTM_FCN",
        "d3m.primitives.feature_extraction.nk_sent2vec.Sent2Vec",
        "d3m.primitives.remote_sensing.mlp.MlpClassifier",
        "d3m.primitives.similarity_modeling.iterative_labeling.ImageRetrieval",
    ]

    prims_to_pipelines = {
        "d3m.primitives.data_cleaning.column_type_profiler.Simon": [
            (SimonPipeline(), ('185_baseball_MIN_METADATA',))
        ],
        "d3m.primitives.data_cleaning.geocoding.Goat_forward": [
            (GoatForwardPipeline(), ('LL0_acled_reduced_MIN_METADATA',))
        ],
        "d3m.primitives.data_cleaning.geocoding.Goat_reverse": [
            (GoatReversePipeline(), ('LL0_acled_reduced_MIN_METADATA',))
        ],
        "d3m.primitives.feature_extraction.nk_sent2vec.Sent2Vec": [
            (Sent2VecPipeline(), ('LL1_TXT_CLS_apple_products_sentiment_MIN_METADATA',))
        ],
        "d3m.primitives.clustering.k_means.Sloth": [
            (StorcPipeline(), ('66_chlorineConcentration_MIN_METADATA',))
        ],
        "d3m.primitives.clustering.hdbscan.Hdbscan": [
            (HdbscanPipeline(), ('SEMI_1044_eye_movements_MIN_METADATA',))
        ],
        "d3m.primitives.clustering.spectral_graph.SpectralClustering": [
            (SpectralClusteringPipeline(), ('SEMI_1044_eye_movements_MIN_METADATA',))
        ],
        "d3m.primitives.dimensionality_reduction.t_distributed_stochastic_neighbor_embedding.Tsne": [
            (TsnePipeline(), ('SEMI_1044_eye_movements_MIN_METADATA',))
        ],
        "d3m.primitives.time_series_classification.k_neighbors.Kanine": [
            (KaninePipeline(), ('66_chlorineConcentration_MIN_METADATA',))
        ],
        "d3m.primitives.time_series_classification.convolutional_neural_net.LSTM_FCN": [
            (LstmFcnPipeline(), (
                '66_chlorineConcentration_MIN_METADATA',
                "LL1_Adiac_MIN_METADATA",
                "LL1_ArrowHead_MIN_METADATA",
                "LL1_Cricket_Y_MIN_METADATA",
                "LL1_ECG200_MIN_METADATA",
                "LL1_ElectricDevices_MIN_METADATA",
                "LL1_FISH_MIN_METADATA",
                "LL1_FaceFour_MIN_METADATA",
                "LL1_HandOutlines_MIN_METADATA",
                "LL1_Haptics_MIN_METADATA",
                "LL1_ItalyPowerDemand_MIN_METADATA",
                "LL1_Meat_MIN_METADATA",
                "LL1_OSULeaf_MIN_METADATA",
            )),
            (LstmFcnPipeline(attention_lstm=True), (
                '66_chlorineConcentration_MIN_METADATA',
                "LL1_Adiac_MIN_METADATA",
                "LL1_ArrowHead_MIN_METADATA",
                "LL1_Cricket_Y_MIN_METADATA",
                "LL1_ECG200_MIN_METADATA",
                "LL1_ElectricDevices_MIN_METADATA",
                "LL1_FISH_MIN_METADATA",
                "LL1_FaceFour_MIN_METADATA",
                "LL1_HandOutlines_MIN_METADATA",
                "LL1_Haptics_MIN_METADATA",
                "LL1_ItalyPowerDemand_MIN_METADATA",
                "LL1_Meat_MIN_METADATA",
                "LL1_OSULeaf_MIN_METADATA",
            ))
        ],
        "d3m.primitives.time_series_forecasting.vector_autoregression.VAR": [
            (VarPipeline(), (
                '56_sunspots_MIN_METADATA',
                '56_sunspots_monthly_MIN_METADATA',
                'LL1_736_population_spawn_MIN_METADATA',
                'LL1_736_stock_market_MIN_METADATA',
                'LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA',
                "LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA",
                "LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA",
                "LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA",
                'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA',
                'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA',
                'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA',
            ))
        ],
        "d3m.primitives.time_series_forecasting.lstm.DeepAR": [
            (DeepARPipeline(prediction_length = 21, context_length = 21), ('56_sunspots_MIN_METADATA',)), 
            (DeepARPipeline(prediction_length = 38, context_length = 38), ('56_sunspots_monthly_MIN_METADATA',)),
            (DeepARPipeline(prediction_length = 60, context_length = 30), ('LL1_736_population_spawn_MIN_METADATA',)),
            (DeepARPipeline(prediction_length = 34, context_length = 17), ('LL1_736_stock_market_MIN_METADATA',)),
        ],
        "d3m.primitives.time_series_forecasting.feed_forward_neural_net.NBEATS": [
            (NBEATSPipeline(prediction_length = 21), ('56_sunspots_MIN_METADATA',)), 
            (NBEATSPipeline(prediction_length = 38), ('56_sunspots_monthly_MIN_METADATA',)),
            (NBEATSPipeline(prediction_length = 60), ('LL1_736_population_spawn_MIN_METADATA',)),
            (NBEATSPipeline(prediction_length = 34), ('LL1_736_stock_market_MIN_METADATA',)),
        ],
        "d3m.primitives.object_detection.retina_net.ObjectDetectionRN": [
            (ObjectDetectionRNPipeline(), (
                'LL1_tidy_terra_panicle_detection_MIN_METADATA', 
                'LL1_penn_fudan_pedestrian_MIN_METADATA'
            ))
        ],
        "d3m.primitives.data_cleaning.data_cleaning.Datacleaning": [
            (DataCleaningPipeline(), ('185_baseball_MIN_METADATA',))
        ],
        "d3m.primitives.data_cleaning.text_summarization.Duke": [
            (DukePipeline(), ('185_baseball_MIN_METADATA',))
        ],
        "d3m.primitives.feature_selection.pca_features.Pcafeatures": [
            (PcaFeaturesPipeline(), ('185_baseball_MIN_METADATA',))
        ],
        "d3m.primitives.feature_selection.rffeatures.Rffeatures": [
            (RfFeaturesPipeline(), ('185_baseball_MIN_METADATA',))
        ],
        "d3m.primitives.classification.inceptionV3_image_feature.Gator": [
            (GatorPipeline(), (
                "124_174_cifar10_MIN_METADATA",
                "124_188_usps_MIN_METADATA",
                "124_214_coil20_MIN_METADATA",
                "uu_101_object_categories_MIN_METADATA",
            ))
        ],
        "d3m.primitives.remote_sensing.mlp.MlpClassifier": [
            (MlpClassifierPipeline(), ('LL1_bigearth_landuse_detection',))
        ],
        "d3m.primitives.similarity_modeling.iterative_labeling.ImageRetrieval": [
            (MlpClassifierPipeline('LL1_bigearth_landuse_detection'), ())
        ]
    }

    for primitive, pipelines in prims_to_pipelines.items():

        if gpu:
            if primitive not in gpu_prims:
                continue
        else:
            if primitive in gpu_prims:
                continue

        os.chdir(f'/annotations/{primitive}')
        os.chdir(os.listdir('.')[0])
        if not os.path.isdir('pipelines'):
            os.mkdir('pipelines')
        else:
            [os.remove(f'pipelines/{pipeline}') for pipeline in os.listdir('pipelines')]
        if not os.path.isdir('pipeline_runs'):
            os.mkdir('pipeline_runs')
        else:
            [os.remove(f'pipeline_runs/{pipeline_run}') for pipeline_run in os.listdir('pipeline_runs')]
        if not os.path.isdir(f'/pipeline_scores/{primitive.split(".")[-1]}'):
            os.mkdir(f'/pipeline_scores/{primitive.split(".")[-1]}')
        for pipeline, datasets in pipelines:
            pipeline.write_pipeline(output_dir = './pipelines')
            for dataset in datasets:
                print(f'Generating pipeline for {primitive.split(".")[-1]} on {dataset}')
                if primitive.split(".")[-1] in ['Duke', 'Sloth']:
                    pipeline.fit_produce(
                        dataset,
                        output_yml_dir = './pipeline_runs',
                        submission = True
                    )
                else:
                    if primitive.split(".")[-1] == 'NBEATS':
                        shutil.rmtree(f'/scratch_dir/nbeats')
                    pipeline.fit_score(
                        dataset,
                        output_yml_dir = './pipeline_runs',
                        output_score_dir = f'/pipeline_scores/{primitive.split(".")[-1]}',
                        submission = True
                    )
        os.system('gzip -r pipeline_runs')

if __name__ == '__main__':
    fire.Fire(generate_pipelines)