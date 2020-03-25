#!/bin/bash -e

#cd /primitives
#git checkout simon_pipelines
#git remote add upstream https://gitlab.com/datadrivendiscovery/primitives
#git pull upstream master

cd /primitives
#git checkout pca_prim_base
#git pull upstream master
#Datasets=('30_personae' '196_autoMpg' '26_radon_seed' '38_sick' '4550_miceProtein' 'LL0_207_autoPrice' '57_hypothyroid' 'LL0_acled_reduced')
Datasets=('196_autoMpg' '26_radon_seed' 'LL0_207_autoPrice')
cd /primitives/v2019.6.7/Distil/d3m.primitives.feature_selection.pca_features.Pcafeatures/3.0.2/pipelines
#mkdir test_pipeline
#mkdir experiments
# create text file to record scores and timing information
#touch scores.txt
#echo "DATASET, SCORE, THRESHOLD, ONLY_NUMERIC_COLS, EXECUTION TIME" >> scores.txt
cd test_pipeline

file="/src/pcafeaturesd3mwrapper/PcafeaturesD3MWrapper/python_pipeline_generator_pcafeatures.py"
match="step_3.add_output('produce')"
insert="temporary line threshold"
sed -i "s/$match/$match\n$insert/" $file
insert="temporary line only_numeric_cols"
sed -i "s/$match/$match\n$insert/" $file

for i in "${Datasets[@]}"; do
  best_score=1000000
  for n in $(seq 0 0.1 0.15); do
    file="/src/pcafeaturesd3mwrapper/PcafeaturesD3MWrapper/python_pipeline_generator_pcafeatures.py"
    sed -i '/threshold/d' $file
    sed -i '/only_numeric_cols/d' $file
    insert="step_3.add_hyperparameter(name='threshold', argument_type=ArgumentType.VALUE,data=$n)"
    sed -i "s/$match/$match\n$insert/" $file
    # generate and save pipeline + metafile
    python3 "/src/pcafeaturesd3mwrapper/PcafeaturesD3MWrapper/python_pipeline_generator_pcafeatures.py" $i

    # test and score pipeline
    start=`date +%s`
    python3 -m d3m runtime -d /datasets/ fit-score -m *.meta -p *.json -c scores.csv
    end=`date +%s`
    runtime=$((end-start))

    # copy pipeline if execution time is less than one hour
    if [ $runtime -lt 3600 ]; then
      echo "$i took less than 1 hour, evaluating pipeline"
      IFS=, read col1 score col3 col4 < <(tail -n1 scores.csv)
      echo "$score"
      echo "$i, $score, $n, True, $runtime" >> ../scores.txt
      cp *.meta ../experiments/
      cp *.json ../experiments/
    fi

    # cleanup temporary file
    rm *.meta
    rm *.json
    rm scores.csv

    insert="step_3.add_hyperparameter(name='only_numeric_cols', argument_type=ArgumentType.VALUE,data=False)"
    sed -i "s/$match/$match\n$insert/" $file
    # generate and save pipeline + metafile
    python3 "/src/pcafeaturesd3mwrapper/PcafeaturesD3MWrapper/python_pipeline_generator_pcafeatures.py" $i

    # test and score pipeline
    start=`date +%s`
    python3 -m d3m runtime -d /datasets/ fit-score -m *.meta -p *.json -c scores.csv
    end=`date +%s`
    runtime=$((end-start))

    # copy pipeline if execution time is less than one hour
    if [ $runtime -lt 3600 ]; then
      echo "$i took less than 1 hour, evaluating pipeline"
      IFS=, read col1 score col3 col4 < <(tail -n1 scores.csv)
      echo "$score"
      echo "$i, $score, $n, False, $runtime" >> ../scores.txt
      cp *.meta ../experiments/
      cp *.json ../experiments/
    fi

    # cleanup temporary file
    rm *.meta
    rm *.json
    rm scores.csv
  done
done