import unittest
import os
import pandas as pd
from gluonts.distribution import NegativeBinomialOutput, StudentTOutput

from primitives.ts_forecasting.deep_ar.deepar import DeepArPrimitive
from d3m.primitives.schema_discovery import profiler
from d3m.primitives.data_transformation import column_parser, extract_columns_by_semantic_types, grouping_field_compose

import utils as test_utils

class PreProcessPipeline():

    # imputer?
    def __init__(self, group_compose = False):

        profiler_hp = profiler.Common.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        parser_hp = column_parser.Common.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        extract_hp = extract_columns_by_semantic_types.Common.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        group_hp = grouping_field_compose.Common.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']

        self.prof = profiler.Common(hyperparams = profiler_hp.defaults())
        self.parse = column_parser.Common(
            hyperparams = parser_hp(
                parser_hp.defaults(),
                parse_semantic_types = [
                    "http://schema.org/Boolean",
                    "http://schema.org/Integer",
                    "http://schema.org/Float",
                    "https://metadata.datadrivendiscovery.org/types/FloatVector",
                    "http://schema.org/DateTime",
                ]
            )
        )
        self.group = grouping_field_compose.Common(hyperparams = group_hp.defaults())
        self.extract_attr = extract_columns_by_semantic_types.Common(
            hyperparams = extract_hp(
                extract_hp.defaults(),
                semantic_types = [
                    "https://metadata.datadrivendiscovery.org/types/Attribute",
                    "https://metadata.datadrivendiscovery.org/types/GroupingKey"
                ]
            )
        )
        self.extract_targ = extract_columns_by_semantic_types.Common(
            hyperparams = extract_hp(
                extract_hp.defaults(),
                semantic_types = ["https://metadata.datadrivendiscovery.org/types/TrueTarget"]
            )
        )
        self.group_compose = group_compose

    def fit(self, df):
        self.prof.set_training_data(inputs = df)
        self.prof.fit()

    def produce(self, df):
        df = self.prof.produce(inputs = df).value
        df = self.parse.produce(inputs = df).value
        if self.group_compose:
            df = self.group.produce(inputs = df).value
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
        '56_sunspots_MIN_METADATA': '12M',
        '56_sunspots_monthly_MIN_METADATA': 'M', 
        'LL1_736_population_spawn_MIN_METADATA': 'D',
        'LL1_736_stock_market_MIN_METADATA': 'D',
        'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA': 'D',
        'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA': 'M',
        'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA': 'W'
    }
    min_pred_lengths = {
        '56_sunspots_MIN_METADATA': 21,
        '56_sunspots_monthly_MIN_METADATA': 38, 
        'LL1_736_population_spawn_MIN_METADATA': 60,
        'LL1_736_stock_market_MIN_METADATA': 34,
        'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA': 100,
        'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA': 26,
        'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA': 30
    }
    grouping_cols = {
        '56_sunspots_MIN_METADATA': [],
        '56_sunspots_monthly_MIN_METADATA': [], 
        'LL1_736_population_spawn_MIN_METADATA': [0,1],
        'LL1_736_stock_market_MIN_METADATA': [0],
        'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA': [0,1],
        'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA': [0,1,2],
        'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA': [0,1,2],
    }
    real_cols = {
        '56_sunspots_MIN_METADATA': [1,2],
        '56_sunspots_monthly_MIN_METADATA': [], 
        'LL1_736_population_spawn_MIN_METADATA': [],
        'LL1_736_stock_market_MIN_METADATA': [],
        'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA': [],
        'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA': [],
        'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA': [],
    }
    distr = {
        '56_sunspots_MIN_METADATA': StudentTOutput,
        '56_sunspots_monthly_MIN_METADATA': StudentTOutput, 
        'LL1_736_population_spawn_MIN_METADATA': NegativeBinomialOutput,
        'LL1_736_stock_market_MIN_METADATA': StudentTOutput,
        'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA': StudentTOutput,
        'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA': NegativeBinomialOutput,
        'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA': NegativeBinomialOutput
    }

    # def test_sunspots(self):        
    #     self._test_ts('56_sunspots_MIN_METADATA', 4)

    # def test_sunspots_monthly(self):        
    #     self._test_ts('56_sunspots_monthly_MIN_METADATA', 2)

    # def test_pop_spawn(self):        
    #     self._test_ts('LL1_736_population_spawn_MIN_METADATA', 4)
        # self._test_ts('LL1_736_population_spawn_MIN_METADATA', 4, group_compose=True)

    # def test_stock(self):        
    #     self._test_ts('LL1_736_stock_market_MIN_METADATA', 3)
        # self._test_ts('LL1_736_stock_market_MIN_METADATA', 3, group_compose=True)

    # def test_terra_leaf(self):        
    #     self._test_ts('LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA', 4)
    #     self._test_ts('LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA', 4, group_compose=True)
    
    def test_phem_monthly(self):        
        self._test_ts('LL1_PHEM_Monthly_Malnutrition_MIN_METADATA', 5)
    #     self._test_ts('LL1_PHEM_Monthly_Malnutrition_MIN_METADATA', 5, group_compose=True)

    # def test_phem_weekly(self):        
    #     self._test_ts('LL1_PHEM_weeklyData_malnutrition_MIN_METADATA', 5)
    #     self._test_ts('LL1_PHEM_weeklyData_malnutrition_MIN_METADATA', 5, group_compose=True)

    def _test_ts(self, dataset_name, target_col, group_compose = False, quantiles = [0.1,0.9]):
        
        # train
        dataset = test_utils.load_dataset(f'/datasets/seed_datasets_current/{dataset_name}/TRAIN/dataset_TRAIN')
        df = test_utils.get_dataframe(dataset, 'learningData', target_col)
        preprocess = PreProcessPipeline(group_compose=group_compose)
        preprocess.fit(df)
        inputs, outputs = preprocess.produce(df)

        deepar_hp = DeepArPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        deepar = DeepArPrimitive(
            hyperparams = deepar_hp(
                deepar_hp.defaults(),
                epochs = 1,
                steps_per_epoch = 1,
                number_samples = 10,
                prediction_length = self.min_pred_lengths[dataset_name],
                context_length = self.min_pred_lengths[dataset_name],
                quantiles = quantiles,
                output_mean = False
            )
        )

        # ------------ PARSING CHECK ------------
        deepar.set_training_data(inputs=inputs, outputs=outputs)
        if group_compose:
            self.assertEqual(deepar._grouping_columns, [inputs.shape[1]-1])
        else:
            self.assertEqual(self.grouping_cols[dataset_name], deepar._grouping_columns)
        self.assertEqual(self.freqs[dataset_name], deepar._freq)
        self.assertEqual(self.real_cols[dataset_name], deepar._real_columns)
        self.assertIsInstance(deepar._deepar_dataset.get_distribution_type(), self.distr[dataset_name]) 

        # ------------ TRAIN PREDS CHECK ------------
        deepar.fit()
        # preds = deepar.produce(inputs = inputs).value
        # self.assertEqual(preds.shape[0], inputs.shape[0]) 

        # ------------ TEST PREDS CHECK ------------        
        dataset = test_utils.load_dataset(f'/datasets/seed_datasets_current/{dataset_name}/TEST/dataset_TEST/')
        df = test_utils.get_dataframe(dataset, 'learningData', target_col)
        inputs, _ = preprocess.produce(df)
        preds = deepar.produce(inputs = inputs).value
        self.assertEqual(preds.shape[0], inputs.shape[0])  

        # ------------ TEST CINTS CHECK ------------        
        # confidence_intervals = deepar.produce_confidence_intervals(inputs = inputs).value
        # self.assertEqual(confidence_intervals.shape[0], inputs.shape[0]) 
        # self.assertEqual(confidence_intervals.shape[1], len(quantiles) + 1)
        # self.assertTrue((confidence_intervals.iloc[:, 1] <= confidence_intervals.iloc[:, 0]).all())
        # self.assertTrue((confidence_intervals.iloc[:, 2] >= confidence_intervals.iloc[:, 0]).all())

if __name__ == '__main__':
    unittest.main()
