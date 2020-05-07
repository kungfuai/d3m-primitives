#!/bin/bash -e 

#######
#
# Usage
#
# ./pull-datasets.sh 
#
# Add datasets you would like to pull from git lfs to Datasets array
#   - only one space between strings in Datasets array, no other separators!
#
#######

cd datasets

#Datasets=('uu_101_object_categories_MIN_METADATA' '56_sunspots_MIN_METADATA' '56_sunspots_monthly_MIN_METADATA' 'LL1_736_population_spawn_MIN_METADATA' 'LL1_736_population_spawn_simpler_MIN_METADATA' 'LL1_736_stock_market_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA' 'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA' 'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA' 'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA' '185_baseball_MIN_METADATA' '1491_one_hundred_plants_margin_MIN_METADATA' 'LL0_1100_popularkids_MIN_METADATA' '38_sick_MIN_METADATA' '4550_MiceProtein_MIN_METADATA' '57_hypothyroid_MIN_METADATA' 'LL0_acled_reduced_MIN_METADATA' 'SEMI_1040_sylva_prior_MIN_METADATA' 'SEMI_1044_eye_movements_MIN_METADATA' 'SEMI_1217_click_prediction_small_MIN_METADATA')
#Datasets=('56_sunspots_MIN_METADATA' '56_sunspots_monthly_MIN_METADATA' 'LL1_736_population_spawn_MIN_METADATA' 'LL1_736_population_spawn_simpler_MIN_METADATA' 'LL1_736_stock_market_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA' 'LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA' 'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA' 'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA' 'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA') 
Datasets=('LL1_PHEM_weeklyData_malnutrition_MIN_METADATA')

for i in "${Datasets[@]}"; do
    git lfs pull -I "seed_datasets_current/$i/"
done

