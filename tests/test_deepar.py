import os
import subprocess

import pandas as pd
from gluonts.distribution import NegativeBinomialOutput, StudentTOutput
from d3m.primitives.schema_discovery import profiler
from d3m.primitives.data_transformation import column_parser, extract_columns_by_semantic_types, grouping_field_compose

from kf_d3m_primitives.ts_forecasting.deep_ar.deepar import DeepArPrimitive
from kf_d3m_primitives.ts_forecasting.deep_ar.deepar_pipeline import DeepARPipeline
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

datetime_format_strs = {
    '56_sunspots_MIN_METADATA': '%Y',
    '56_sunspots_monthly_MIN_METADATA': '%Y-%m', 
    'LL1_736_population_spawn_MIN_METADATA': '%j',
    'LL1_736_stock_market_MIN_METADATA': '%m/%d/%Y',
    'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA': '%j',
    'LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA': '%j',
    'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA': '%Y-%m-%d',
    'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA': '%Y-%m-%d'  
}

freqs = {
    '56_sunspots_MIN_METADATA': '12M',
    '56_sunspots_monthly_MIN_METADATA': 'M', 
    'LL1_736_population_spawn_MIN_METADATA': 'D',
    'LL1_736_stock_market_MIN_METADATA': 'D',
    'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA': 'D',
    'LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA': 'D',
    'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA': 'M',
    'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA': 'W'
}
min_pred_lengths = {
    '56_sunspots_MIN_METADATA': (21, 40),
    '56_sunspots_monthly_MIN_METADATA': (38, 321), 
    'LL1_736_population_spawn_MIN_METADATA': (60, 100),
    'LL1_736_stock_market_MIN_METADATA': (34, 50),
    'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA': (30, 45),
    'LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA': (30, 45),
    'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA': (10, 15),
    'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA': (10, 15)
}
grouping_cols = {
    '56_sunspots_MIN_METADATA': [],
    '56_sunspots_monthly_MIN_METADATA': [], 
    'LL1_736_population_spawn_MIN_METADATA': [0,1],
    'LL1_736_stock_market_MIN_METADATA': [0],
    'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA': [0,1],
    'LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA': [0,1],
    'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA': [0,1,2],
    'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA': [0,1,2],
}
real_cols = {
    '56_sunspots_MIN_METADATA': [1,2],
    '56_sunspots_monthly_MIN_METADATA': [], 
    'LL1_736_population_spawn_MIN_METADATA': [],
    'LL1_736_stock_market_MIN_METADATA': [],
    'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA': [],
    'LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA': [],
    'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA': [],
    'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA': [],
}
distr = {
    '56_sunspots_MIN_METADATA': StudentTOutput,
    '56_sunspots_monthly_MIN_METADATA': StudentTOutput, 
    'LL1_736_population_spawn_MIN_METADATA': NegativeBinomialOutput,
    'LL1_736_stock_market_MIN_METADATA': StudentTOutput,
    'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA': StudentTOutput,
    'LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA': NegativeBinomialOutput,
    'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA': NegativeBinomialOutput,
    'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA': StudentTOutput
}

def _test_set_training_data(dataset_name, target_col, group_compose = False, split_train = False):
    dataset = test_utils.load_dataset(f'/datasets/seed_datasets_current/{dataset_name}/TRAIN/dataset_TRAIN')
    df = test_utils.get_dataframe(dataset, 'learningData', target_col)
    time_col = df.metadata.list_columns_with_semantic_types(
        (
            "https://metadata.datadrivendiscovery.org/types/Time",
            "http://schema.org/DateTime",
        )
    )[0]
    original_times = df.iloc[:, time_col]
    df.iloc[:, time_col] = pd.to_datetime(
        df.iloc[:, time_col], 
        format = datetime_format_strs[dataset_name]
    )
    df = df.sort_values(by = df.columns[time_col])
    df.iloc[:, time_col] = original_times
    train_split = int(0.9 * df.shape[0])
    train = df.iloc[:train_split, :].reset_index(drop=True)
    val = df.iloc[train_split:, :].reset_index(drop=True)
    df = df.reset_index(drop=True)

    preprocess = PreProcessPipeline(group_compose=group_compose)
    preprocess.fit(train)
    train_inputs, train_outputs = preprocess.produce(train)
    val_inputs, _ = preprocess.produce(val)
    all_inputs, all_outputs = preprocess.produce(df)
    
    pred_length_idx = 1 if split_train else 0  
    deepar_hp = DeepArPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'] 
    deepar = DeepArPrimitive(
        hyperparams = deepar_hp(
            deepar_hp.defaults(),
            epochs = 1,
            steps_per_epoch = 1,
            number_samples = 10,
            prediction_length = min_pred_lengths[dataset_name][pred_length_idx] + 5,
            context_length = min_pred_lengths[dataset_name][pred_length_idx] - 5,
            quantiles = (0.1, 0.9),
            output_mean = False
        )
    )
    if split_train:
        deepar.set_training_data(inputs=train_inputs, outputs=train_outputs)
    else:
        deepar.set_training_data(inputs=all_inputs, outputs=all_outputs)
    
    if group_compose:
        assert deepar._grouping_columns == [train_inputs.shape[1]-1]
    else:
        assert grouping_cols[dataset_name] == deepar._grouping_columns
    assert freqs[dataset_name] == deepar._freq
    assert real_cols[dataset_name] == deepar._real_columns
    assert isinstance(deepar._deepar_dataset.get_distribution_type(), distr[dataset_name])
    deepar.fit()
    return deepar, preprocess, train_inputs, val_inputs, all_inputs

def _test_produce_train_data(deepar, train_inputs, val_inputs, all_inputs):
    train_preds = deepar.produce(inputs = train_inputs).value
    assert train_preds.shape[0] == train_inputs.shape[0] 
    val_preds = deepar.produce(inputs = val_inputs).value
    assert val_preds.shape[0] == val_inputs.shape[0] 
    all_preds = deepar.produce(inputs = all_inputs).value
    assert all_preds.shape[0] == all_inputs.shape[0] 

def _test_produce_test_data(deepar, inputs_test):
    test_preds = deepar.produce(inputs = inputs_test).value
    assert test_preds.shape[0] == inputs_test.shape[0] 

def _test_produce_confidence_intervals(deepar, inputs):
    confidence_intervals = deepar.produce_confidence_intervals(inputs = inputs).value
    assert confidence_intervals.shape[0] == inputs.shape[0]
    assert confidence_intervals.shape[1] == 3
    assert (confidence_intervals.iloc[:, 1].dropna() <= confidence_intervals.iloc[:, 0].dropna()).all()
    assert (confidence_intervals.iloc[:, 2].dropna() >= confidence_intervals.iloc[:, 0].dropna()).all()

def _test_ts(dataset_name, target_col, group_compose = False, split_train = False):
    deepar, preprocess, inputs_train, inputs_val, inputs_all = _test_set_training_data(
        dataset_name,
        target_col, 
        group_compose=group_compose,
        split_train=split_train
    )
    _test_produce_train_data(deepar, inputs_train, inputs_val, inputs_all)

    dataset = test_utils.load_dataset(f'/datasets/seed_datasets_current/{dataset_name}/TEST/dataset_TEST/')
    df = test_utils.get_dataframe(dataset, 'learningData', target_col)
    inputs_test, _ = preprocess.produce(df)
    
    _test_produce_test_data(deepar, inputs_test)
    _test_produce_confidence_intervals(deepar, inputs_all)
    _test_produce_confidence_intervals(deepar, inputs_test)

def _test_serialize(dataset, group_compose = False):
    
    pipeline = DeepARPipeline(
        epochs = 1,
        steps_per_epoch = 1,
        number_samples = 10,
        prediction_length = min_pred_lengths[dataset][0],
        context_length = min_pred_lengths[dataset][0],
        group_compose = group_compose,
    )
    pipeline.write_pipeline()
    pipeline.fit_serialize(dataset)
    pipeline.deserialize_score(dataset)
    pipeline.delete_pipeline()
    pipeline.delete_serialized_pipeline()

def _test_confidence_intervals(dataset, group_compose = False):
    
    pipeline = DeepARPipeline(
        epochs = 1,
        steps_per_epoch = 1,
        number_samples = 10,
        prediction_length = min_pred_lengths[dataset][0],
        context_length = min_pred_lengths[dataset][0],
        group_compose = group_compose,
        confidence_intervals = True
    )    
    pipeline.write_pipeline()
    pipeline.fit_produce(dataset)
    pipeline.delete_pipeline()

# def test_fit_produce_dataset_sunspots():
#     _test_ts('56_sunspots_MIN_METADATA', 4)

# def test_fit_produce_dataset_sunspots_monthly():
#     _test_ts('56_sunspots_monthly_MIN_METADATA', 2)

# def test_fit_produce_dataset_stock():        
#     _test_ts('LL1_736_stock_market_MIN_METADATA', 3)

# def test_fit_produce_dataset_pop_spawn():
#     _test_ts('LL1_736_population_spawn_MIN_METADATA', 4)

# def test_fit_produce_dataset_terra():      
#     _test_ts('LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA', 4)

# def test_fit_produce_dataset_terra_80():      
#     _test_ts('LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA', 4)

# def test_fit_produce_dataset_phem_monthly():     
#     _test_ts('LL1_PHEM_Monthly_Malnutrition_MIN_METADATA', 5)

# def test_fit_produce_dataset_phem_weekly():     
#     _test_ts('LL1_PHEM_weeklyData_malnutrition_MIN_METADATA', 5)

# def test_fit_produce_split_dataset_sunspots():
#     _test_ts('56_sunspots_MIN_METADATA', 4, split_train=True)

# def test_fit_produce_split_dataset_sunspots_monthly():
#     _test_ts('56_sunspots_monthly_MIN_METADATA', 2, split_train=True)

# def test_fit_produce_split_dataset_stock():        
#     _test_ts('LL1_736_stock_market_MIN_METADATA', 3, split_train=True)

# def test_fit_produce_split_dataset_pop_spawn():
#     _test_ts('LL1_736_population_spawn_MIN_METADATA', 4, group_compose=True, split_train=True)

# def test_fit_produce_split_dataset_terra():      
#     _test_ts('LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA', 4, group_compose = True, split_train=True)

# def test_fit_produce_split_dataset_phem_monthly():     
#     _test_ts('LL1_PHEM_Monthly_Malnutrition_MIN_METADATA', 5, group_compose = True, split_train=True)

# def test_fit_produce_split_dataset_phem_weekly():     
#     _test_ts('LL1_PHEM_weeklyData_malnutrition_MIN_METADATA', 5, group_compose = True, split_train=True)

def test_serialization_dataset_sunspots():
    _test_serialize('56_sunspots_MIN_METADATA')

def test_serialization_dataset_sunspots_monthly():
    _test_serialize('56_sunspots_monthly_MIN_METADATA')

def test_serialization_dataset_pop_spawn():
    _test_serialize('LL1_736_population_spawn_MIN_METADATA')

def test_serialization_dataset_stock():
    _test_serialize('LL1_736_stock_market_MIN_METADATA')

def test_confidence_intervals_dataset_sunspots():
    _test_confidence_intervals('56_sunspots_MIN_METADATA')

def test_confidence_intervals_dataset_sunspots_monthly():
    _test_confidence_intervals('56_sunspots_monthly_MIN_METADATA')

def test_confidence_intervals_dataset_pop_spawn():
    _test_confidence_intervals('LL1_736_population_spawn_MIN_METADATA', group_compose=True)

def test_confidence_intervals_dataset_stock():
    _test_confidence_intervals('LL1_736_stock_market_MIN_METADATA')

