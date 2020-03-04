#!/bin/bash -e 

Datasets=('124_174_cifar10' '124_188_usps' '124_214_coil20' 'uu_101_object_categories')
cd /primitives/v2019.11.10/Distil/d3m.primitives.digital_image_processing.convolutional_neural_net.Gator/1.0.2/pipeline_runs

#create text file to record scores and timing information
touch scores.txt
echo "DATASET, SCORE, EXECUTION TIME" >> scores.txt

for i in "${Datasets[@]}"; do

  start=`date +%s`
  python3 -m d3m runtime -v /src/gator/gator -d /datasets/ fit-score -p ../pipelines/*.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -a /datasets/seed_datasets_current/$i/SCORE/dataset_SCORE/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -c scores.csv -O ${i}_validation.yml
  end=`date +%s`
  runtime=$((end-start))

  echo "----------$i took $runtime----------"

  # save information
  IFS=, read col1 score col3 col4 < <(tail -n1 scores.csv)
  echo "$i, $score, $runtime" >> scores.txt
  
  # # cleanup temporary file
  rm scores.csv
done

docker run --rm -t -i --mount type=bind,source=/Users/jgleason/Documents/NewKnowledge/D3M/gator,target=/gator --mount type=bind,source=/Users/jgleason/Documents/NewKnowledge/D3M/datasets,target=/datasets registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9