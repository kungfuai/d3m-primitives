import unittest
import os
import pandas as pd
from gluonts.distribution import NegativeBinomialOutput, StudentTOutput

from d3m.primitives.time_series_forecasting.lstm import DeepAR
from d3m.primitives.schema_discovery import profiler
from d3m.primitives.data_transformation import column_parser, extract_columns_by_semantic_types, grouping_field_compose

import utils as test_utils

class PreProcessPipeline():

    # imputer?
    # gf compose?
    def __init__(self):

        profiler_hp = profiler.Common.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        parser_hp = column_parser.Common.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        extract_hp = \
            extract_columns_by_semantic_types.Common.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']

        self.prof = profiler.Common(hyperparams = profiler_hp.defaults())
        self.parse = column_parser.Common(hyperparams = parser_hp.defaults())
        self.extract_attr = extract_columns_by_semantic_types.Common(
            hyperparams = extract_hp(
                extract_hp.defaults(),
                semantic_types = ["https://metadata.datadrivendiscovery.org/types/Attribute"]
            )
        )
        self.extract_targ = extract_columns_by_semantic_types.Common(
            hyperparams = extract_hp(
                extract_hp.defaults(),
                semantic_types = ["https://metadata.datadrivendiscovery.org/types/TrueTarget"]
            )
        )

    def fit_produce(self, df):
        self.prof.set_training_data(inputs = df)
        self.prof.fit()
        df = self.prof.produce(inputs = df).value
        df = self.parse.produce(inputs = df).value
        return (
            self.extract_attr.produce(inputs = df).value, 
            self.extract_targ.produce(inputs = df).value
        )
    def produce(self, df):
        df = self.prof.produce(inputs = df).value
        df = self.parse.produce(inputs = df).value
        return (
            self.extract_attr.produce(inputs = df).value, 
            self.extract_targ.produce(inputs = df).value
        )

class DeepArPrimitiveTestCase(unittest.TestCase):
    """ for each time series dataset want to check:

            -grouping columns correctly identified
            -additional real/cat columns correctly identified
            -count data correctly identified
            -frequency correctly identified
            -identified prediction intervals same length as test frame
            -shape of predictions make sense before slicing on in-sample and test frame        
    """

    freqs = {
        '56_sunspots_MIN_METADATA': 'YS',
        '56_sunspots_monthly_MIN_METADATA': 'MS', 
        'LL1_736_population_spawn_MIN_METADATA': 'D',
        'LL1_736_stock_market_MIN_METADATA': 'D',
        'LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA': 'D', 
        'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA': 'D',
        'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA': 'M',
        'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA': 'W'
    }
    grouping_cols = {
        '56_sunspots_MIN_METADATA': [],
        '56_sunspots_monthly_MIN_METADATA': [], 
        'LL1_736_population_spawn_MIN_METADATA': [1,2],
        'LL1_736_stock_market_MIN_METADATA': [1],
        'LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA': [1,2], 
        'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA': [1,2],
        'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA': [1,2,3],
        'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA': [1,2,3],
    }
    real_cols = {
        '56_sunspots_MIN_METADATA': [1,2],
        '56_sunspots_monthly_MIN_METADATA': [], 
        'LL1_736_population_spawn_MIN_METADATA': [1,2],
        'LL1_736_stock_market_MIN_METADATA': [1],
        'LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA': [1,2], 
        'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA': [1,2],
        'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA': [1,2,3],
        'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA': [1,2,3],
    }
    distr = {
        '56_sunspots_MIN_METADATA': StudentTOutput,
        '56_sunspots_monthly_MIN_METADATA': [], 
        'LL1_736_population_spawn_MIN_METADATA': [1,2],
        'LL1_736_stock_market_MIN_METADATA': [1],
        'LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA': [1,2], 
        'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA': [1,2],
        'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA': [1,2,3],
        'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA': [1,2,3],
    }

    deepar_hp = DeepAR.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']


    def test_sunspots(self):
        
        dataset_name = '56_sunspots_MIN_METADATA'
        dataset = test_utils.load_dataset(f'/datasets/seed_datasets_current/{dataset_name}/{dataset_name}_dataset/')
        df = test_utils.get_dataframe(dataset, 'learningData', 4)
        preprocess = PreProcessPipeline()
        deepar = DeepAR(hyperparams = self.deepar_hp.defaults())

        inputs, outputs = preprocess.fit_produce(df)
        deepar.set_training_data(inputs=inputs, outputs=outputs)

        self.assertEqual(self.grouping_cols[dataset_name], deepar._grouping_columns)
        self.assertEqual(self.freqs[dataset_name], deepar._freq)
        self.assertEqual(self.real_cols[dataset_name], deepar._real_columns)
        self.assertEqual(deepar._cat_columns, []) 
        self.assertIsInstance(deepar._distr_output, self.distr[dataset_name]) 

        # preds on train set: correct shapes
        # preds on test set: correct shapes


    #def test_sun_month(self):
           

if __name__ == '__main__':
    unittest.main()
