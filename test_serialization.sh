#!/bin/bash -e 

#######
#
# Usage
#
# ./test_serialization.sh pipeline dataset
#
# Argument 1: pipeline
# Argument 2: dataset
#
#######

# parse arguments
pipeline=$1
dataset=$2

# python3 -m d3m runtime -d /datasets -v /static_volumes fit-produce -p $pipeline -o test_preds.csv -s test_pipeline.d3m -t /datasets/seed_datasets_current/${dataset}/TEST/dataset_TEST/datasetDoc.json -i /datasets/seed_datasets_current/${dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -r /datasets/seed_datasets_current/${dataset}/${dataset}_problem/problemDoc.json
# head -n 11 test_preds.csv
# rm test_pipeline.d3m
# rm test_preds.csv

# run and save pipeline as binary file `test_pipeline.d3m`
echo 'Running and serializing pipeline'
python3 -m d3m runtime -d /datasets -v /static_volumes fit -p $pipeline -s test_pipeline.d3m -i /datasets/seed_datasets_current/${dataset}/TRAIN/dataset_TRAIN/datasetDoc.json -r /datasets/seed_datasets_current/${dataset}/${dataset}_problem/problemDoc.json

# subsequent produce call that loads saved binary pipeline and writes preds to `test_preds.csv`
echo 'Re-running serialized pipeline'
python3 -m d3m runtime -d /datasets -v /static_volumes produce -f test_pipeline.d3m -o test_preds.csv -t /datasets/seed_datasets_current/${dataset}/TEST/dataset_TEST/datasetDoc.json

echo 'Serialization successful! Here are the first ten predictions:'
head -n 11 test_preds.csv
rm test_pipeline.d3m
rm test_preds.csv
