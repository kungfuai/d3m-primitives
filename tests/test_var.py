from primitives.ts_forecasting.vector_autoregression.var_pipeline import VarPipeline

def _test_serialize(dataset):
    
    pipeline = VarPipeline()
    pipeline.write_pipeline()
    pipeline.fit_serialize(dataset)
    pipeline.deserialize_score(dataset)
    pipeline.delete_pipeline()
    pipeline.delete_serialized_pipeline()
    
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

def test_serialization_dataset_terra_leaf():
    _test_serialize('LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA')

def test_serialization_dataset_phem_monthly():
    _test_serialize('LL1_PHEM_Monthly_Malnutrition_MIN_METADATA')

def test_serialization_dataset_phem_weekly():
    _test_serialize('LL1_PHEM_weeklyData_malnutrition_MIN_METADATA')
