#!/bin/bash -e

### Instructions: Set the name of the pipeline directory your are tesing into
### and then set all the relevant data sets. Note which data sets have an
### inconsistent directory structure as the script will have to be run twice
### to accomodate for both.

pipeline_dir='pipeline_test_files_011420'

# SCORE consistent data sets
Datasets=('185_baseball_MIN_METADATA' '1491_one_hundred_plants_margin_MIN_METADATA' 'LL0_1100_popularkids_MIN_METADATA' '196_autoMpg_MIN_METADATA' '26_radon_seed_MIN_METADATA' '38_sick_MIN_METADATA' '4550_MiceProtein_MIN_METADATA' 'LL0_207_autoPrice_MIN_METADATA' '57_hypothyroid_MIN_METADATA' 'LL0_acled_reduced_MIN_METADATA')

# SCORE inconsistent data sets
#Datasets=('')

# create text file to record scores and timing information
#touch pipeline_tests/scores.txt
#echo "DATASET, SCORE, EXECUTION TIME" >> pipeline_tests/scores.txt

for i in "${Datasets[@]}"; do

    # generate and save pipeline
    python3 "pca_features_pipeline.py" $i

    # test and score pipeline
    start=`date +%s`

    # SCORE consistent pipeline run
    python3 -m d3m runtime -d /datasets/ fit-score -p *.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/$i/SCORE/dataset_SCORE/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -O ${i}_pipeline_run.yaml -c ${i}_scores_con.csv

    # SCORE inconsistent pipeline run
    #python3 -m d3m runtime -d /datasets/ fit-score -p *.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/$i/SCORE/dataset_TEST/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -O ${i}_pipeline_run.yaml -c ${i}_scores_incon.csv

    end=`date +%s`
    runtime=$((end-start))

    #IFS=, read col1 score col3 col4 < <(tail -n1 scores.csv)
    #    echo "$i, $score, $runtime" >> pipeline_tests/scores.txt

    # Cleanup and move files
    mv *.json ${pipeline_dir}/${i}_pipeline.json
    mv *.yaml ${pipeline_dir}/
    mv *.csv ${pipeline_dir}/
    cp pca_features_pipeline.py ${pipeline_dir}/
    #rm *.meta
    #rm scores.csv
done