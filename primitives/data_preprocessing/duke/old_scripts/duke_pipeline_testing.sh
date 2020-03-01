#!/bin/bash -e

### Instructions: Set the name of the pipeline directory your are tesing into
### and then set all the relevant data sets.
### NOTE: DUKE IS FIT-PRODUCE ONLY, NOT FIT-SCORE

pipeline_dir='pipeline_test_files_011420'

# SCORE consistent data sets
#Datasets=('196_autoMpg_MIN_METADATA' '185_baseball_MIN_METADATA' '26_radon_seed_MIN_METADATA' '38_sick_MIN_METADATA' '57_hypothyroid_MIN_METADATA' 'LL0_186_braziltourism_MIN_METADATA' 'LL0_207_autoPrice_MIN_METADATA' 'SEMI_1217_click_prediction_small' 'uu4_SPECT' 'LL0_acled_reduced_MIN_METADATA' 'LL1_multilearn_emotions')
Datasets=('196_autoMpg_MIN_METADATA' '185_baseball_MIN_METADATA' '26_radon_seed_MIN_METADATA' '57_hypothyroid_MIN_METADATA' 'LL0_186_braziltourism_MIN_METADATA' 'LL0_207_autoPrice_MIN_METADATA' 'LL0_acled_reduced_MIN_METADATA')

# SCORE inconsistent data sets
#Datasets=('38_sick_MIN_METADATA')

for i in "${Datasets[@]}"; do

    # generate and save pipeline
    python3 "duke_pipeline.py" $i

    # test and score pipeline
    start=`date +%s`
    python3 -m d3m runtime -d /datasets/ -v /pipeline_data/duke fit-produce -p *.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -O ${i}_pipeline_run.yaml
    end=`date +%s`
    runtime=$((end-start))

    # cleanup temporary file
    mv *.json ${pipeline_dir}/${i}_pipeline.json
    mv *.yaml ${pipeline_dir}/
    cp duke_pipeline.py ${pipeline_dir}/
    rm *.meta
done