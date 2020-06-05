"""
Utility to download D3M datasets from Gitlab. D3M Gitlab credentials are needed
"""

import os
import signal
import subprocess
import time 

if not os.path.isdir('datasets'):
    subprocess.run(
        [
            "GIT_LFS_SKIP_SMUDGE=1",
            "git",
            "clone",
            "--recursive",
            "https://gitlab.datadrivendiscovery.org/d3m/datasets"
        ],
        check = True
    )

datasets = [
    "185_baseball_MIN_METADATA",
    "LL1_736_stock_market_MIN_METADATA",
    "56_sunspots_MIN_METADATA",
    "56_sunspots_monthly_MIN_METADATA",
    "LL1_736_population_spawn_MIN_METADATA",
    "LL1_736_stock_market_MIN_METADATA",
    "124_174_cifar10_MIN_METADATA",
    "124_188_usps_MIN_METADATA",
    "124_214_coil20_MIN_METADATA",
    "uu_101_object_categories_MIN_METADATA",
    "LL0_acled_reduced_MIN_METADATA",
    "SEMI_1044_eye_movements_MIN_METADATA",
    "66_chlorineConcentration_MIN_METADATA",
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
    "LL1_penn_fudan_pedestrian_MIN_METADATA",
    "LL1_tidy_terra_panicle_detection_MIN_METADATA",
    "LL1_TXT_CLS_apple_products_sentiment_MIN_METADATA",
    "LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA",
    "LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA",
    "LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA",
    "LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA",
    "LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA",
    "LL1_PHEM_Monthly_Malnutrition_MIN_METADATA",
    "LL1_PHEM_weeklyData_malnutrition_MIN_METADATA",
]

os.chdir('datasets')
for dataset in datasets:
    proc = subprocess.Popen(
        [
            "git",
            "lfs",
            "pull",
            "-I",
            f"seed_datasets_current/{dataset}/"
        ],
        preexec_fn=os.setsid,
    )
    time.sleep(60)
    if proc.poll() is None: 
        os.killpg(proc.pid, signal.SIGINT)
    print(f'Downloaded dataset: {dataset}')
