#!/bin/bash -e 

Datasets=('185_baseball_MIN_METADATA')
#'1491_one_hundred_plants_margin_MIN_METADATA' 'LL0_1100_popularkids_MIN_METADATA' '38_sick_MIN_METADATA' '4550_MiceProtein_MIN_METADATA' '57_hypothyroid_MIN_METADATA' 'LL0_acled_reduced_MIN_METADATA')
#Datasets=('uu3_world_development_indicators_MIN_METADATA')
# cd /primitives/v2020.1.9/Distil/d3m.primitives.data_cleaning.column_type_profiler.Simon/1.2.2
# cd pipelines
# python3 "/wrap/SimonD3MWrapper/pipelines/simon_pipeline.py"
# cd ../pipeline_runs

#create text file to record scores and timing information
# touch scores.txt
# echo "DATASET, SCORE, EXECUTION TIME" >> scores.txt
for i in "${Datasets[@]}"; do

  # generate pipeline run
  start=`date +%s`
  python3 -m d3m runtime -v /wrap/SimonD3MWrapper -d /datasets/ fit-score -p *.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/$i/SCORE/dataset_SCORE/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -c scores.csv -O $i.yml
  end=`date +%s`
  runtime=$((end-start))

  echo "----------$i took $runtime----------"

  # # save information
  # IFS=, read col1 score col3 col4 < <(tail -n1 scores.csv)
  # echo "$i, $score, $runtime" >> scores.txt
  
  # # # cleanup temporary file
  # rm scores.csv
done

# zip pipeline runs individually
# cd ..
# gzip -r pipeline_runs
