#!/bin/bash -e 

#Datasets=('LL1_336_MS_Geolife_transport_mode_prediction' 'LL1_336_MS_Geolife_transport_mode_prediction_separate_lat' 'LL1_726_TIDY_GPS_carpool_bus_service_rating_prediction' 'LL0_acled_reduced')
Datasets=('LL1_726_TIDY_GPS_carpool_bus_service_rating_prediction' 'LL0_acled_reduced')
#mkdir /primitives/v2019.6.7/Distil/d3m.primitives.data_cleaning.geocoding.Goat_reverse
cd /primitives/v2019.6.7/Distil/d3m.primitives.data_cleaning.geocoding.Goat_reverse
#mkdir 1.0.7
cd 1.0.7
#python3 -m d3m index describe -i 4 d3m.primitives.data_cleaning.geocoding.Goat_reverse > primitive.json
#mkdir pipelines
cd pipelines
#mkdir test_pipeline
cd test_pipeline
rm *
# create text file to record scores and timing information
touch scores.txt
echo "DATASET, SCORE, EXECUTION TIME" >> scores.txt

for i in "${Datasets[@]}"; do

  # generate and save pipeline + metafile
  python3 "/src/goatd3mwrapper/GoatD3MWrapper/goat_reverse_pipeline.py" $i

  # test and score pipeline
  start=`date +%s` 
  python3 -m d3m runtime -d /datasets/ -v /datasets/ fit-score -m *.meta -p *.json -c scores.csv
  end=`date +%s`
  runtime=$((end-start))

  # copy pipeline if execution time is less than one hour
  if [ $runtime -lt 3600 ]; then
     echo "$i took less than 1 hour, copying pipeline"
     cp * ../
  fi

  # save information
  IFS=, read col1 score col3 col4 < <(tail -n1 scores.csv)
  echo "$i, $score, $runtime" >> scores.txt
  
  # cleanup temporary file
  rm *.meta
  rm *.json
  rm scores.csv
done
