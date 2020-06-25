from primitives.ts_forecasting.vector_autoregression.var_pipeline import VarPipeline

def _test_serialize(dataset, group_compose = False):
    
    pipeline = VarPipeline(group_compose=group_compose)
    pipeline.write_pipeline()
    pipeline.fit_serialize(dataset)
    pipeline.deserialize_score(dataset)
    pipeline.delete_pipeline()
    pipeline.delete_serialized_pipeline()

def _test_confidence_intervals(dataset, group_compose = False):
    
    pipeline = VarPipeline(group_compose=group_compose, confidence_intervals=True)
    pipeline.write_pipeline()
    pipeline.fit_produce(dataset)
    pipeline.delete_pipeline()

def _test_confidence_intervals_all(dataset, group_compose = False):
    
    pipeline = VarPipeline(group_compose=group_compose, confidence_intervals=True)
    pipeline.write_pipeline()
    pipeline.fit_produce_all(dataset)
    pipeline.delete_pipeline()

def test_serialization_dataset_sunspots():
    _test_serialize('56_sunspots_MIN_METADATA')

def test_serialization_dataset_sunspots_monthly():
    _test_serialize('56_sunspots_monthly_MIN_METADATA')

def test_serialization_dataset_pop_spawn():
    _test_serialize('LL1_736_population_spawn_MIN_METADATA')

def test_serialization_dataset_stock():
    _test_serialize('LL1_736_stock_market_MIN_METADATA')

def test_serialization_dataset_terra_canopy():
    _test_serialize('LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA')

def test_serialization_dataset_terra_canopy_90():
    _test_serialize('LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA')

def test_serialization_dataset_terra_canopy_80():
    _test_serialize('LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA')

def test_serialization_dataset_terra_canopy_70():
    _test_serialize('LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA')

def test_serialization_dataset_terra_leaf():
    _test_serialize('LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA')

def test_serialization_dataset_phem_monthly():
    _test_serialize('LL1_PHEM_Monthly_Malnutrition_MIN_METADATA')

def test_serialization_dataset_phem_weekly():
    _test_serialize('LL1_PHEM_weeklyData_malnutrition_MIN_METADATA')

# def test_serialization_group_compose_dataset_pop_spawn():
#     _test_serialize('LL1_736_population_spawn_MIN_METADATA', group_compose=True)

# def test_serialization_group_compose_dataset_terra_canopy():
#     _test_serialize('LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA', group_compose=True)

# def test_serialization_group_compose_dataset_terra_canopy_90():
#     _test_serialize('LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA', group_compose=True)

# def test_serialization_group_compose_dataset_terra_canopy_80():
#     _test_serialize('LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA', group_compose=True)

# def test_serialization_group_compose_dataset_terra_canopy_70():
#     _test_serialize('LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA', group_compose=True)

# def test_serialization_group_compose_dataset_terra_leaf():
#     _test_serialize('LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA', group_compose=True)

# def test_serialization_group_compose_dataset_phem_monthly():
#     _test_serialize('LL1_PHEM_Monthly_Malnutrition_MIN_METADATA', group_compose=True)

# def test_serialization_group_compose_dataset_phem_weekly():
#     _test_serialize('LL1_PHEM_weeklyData_malnutrition_MIN_METADATA', group_compose=True)

# def test_confidence_intervals_dataset_sunspots():
#     _test_confidence_intervals('56_sunspots_MIN_METADATA')

# def test_confidence_intervals_dataset_sunspots_monthly():
#     _test_confidence_intervals('56_sunspots_monthly_MIN_METADATA')

# def test_confidence_intervals_dataset_stock():
#     _test_confidence_intervals('LL1_736_stock_market_MIN_METADATA')

# def test_confidence_intervals_dataset_pop_spawn():
#     _test_confidence_intervals('LL1_736_population_spawn_MIN_METADATA')

# def test_confidence_intervals_dataset_terra_canopy():
#     _test_confidence_intervals('LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA')#, group_compose=True)

# def test_confidence_intervals_dataset_terra_canopy_90():
#     _test_confidence_intervals('LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA', group_compose=True)

# def test_confidence_intervals_dataset_terra_canopy_80():
#     _test_confidence_intervals('LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA', group_compose=True)

# def test_confidence_intervals_dataset_terra_canopy_70():
#     _test_confidence_intervals('LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA', group_compose=True)

# def test_confidence_intervals_dataset_terra_leaf():
#     _test_confidence_intervals('LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA', group_compose=True)

# def test_confidence_intervals_dataset_phem_monthly():
#     _test_confidence_intervals('LL1_PHEM_Monthly_Malnutrition_MIN_METADATA', group_compose=True)

# def test_confidence_intervals_dataset_phem_weekly():
#     _test_confidence_intervals('LL1_PHEM_weeklyData_malnutrition_MIN_METADATA', group_compose=True)

# def test_confidence_intervals_all_dataset_sunspots():
#     _test_confidence_intervals_all('56_sunspots_MIN_METADATA')

# def test_confidence_intervals_all_dataset_sunspots_monthly():
#     _test_confidence_intervals_all('56_sunspots_monthly_MIN_METADATA')

# def test_confidence_intervals_all_dataset_stock():
#     _test_confidence_intervals_all('LL1_736_stock_market_MIN_METADATA')

# def test_confidence_intervals_all_dataset_pop_spawn():
#     _test_confidence_intervals_all('LL1_736_population_spawn_MIN_METADATA', group_compose=True)

# def test_confidence_intervals_all_dataset_terra_canopy():
#     _test_confidence_intervals_all('LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA')#, group_compose=True)

# def test_confidence_intervals_all_dataset_terra_canopy_90():
#     _test_confidence_intervals_all('LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA', group_compose=True)

# def test_confidence_intervals_all_dataset_terra_canopy_80():
#     _test_confidence_intervals_all('LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA', group_compose=True)

# def test_confidence_intervals_all_dataset_terra_canopy_70():
#     _test_confidence_intervals_all('LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA', group_compose=True)

# def test_confidence_intervals_all_dataset_terra_leaf():
#     _test_confidence_intervals_all('LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA')

# def test_confidence_intervals_all_dataset_phem_monthly():
#     _test_confidence_intervals_all('LL1_PHEM_Monthly_Malnutrition_MIN_METADATA')

# def test_confidence_intervals_all_dataset_phem_weekly():
#     _test_confidence_intervals_all('LL1_PHEM_weeklyData_malnutrition_MIN_METADATA')
