#!/bin/bash -e

Datasets=('185_baseball' '1491_one_hundred_plants_margin' 'LL0_1100_popularkids' '38_sick' '4550_MiceProtein' '57_hypothyroid' 'LL0_acled_reduced')
Bools=('True' 'False')
cd /primitives/v2019.6.7/Distil/d3m.primitives.feature_selection.rffeatures.Rffeatures/3.1.1/pipelines
git checkout pca_prim_base
#mkdir test_pipeline
#mkdir best_pipelines

# create text file to record scores and timing information
#touch scores.txt
#echo "DATASET, SCORE, FEATURE_PROPORTION, ONLY_NUMERIC, EXECUTION TIME" >> scores.txt
cd test_pipeline

file="/src/rffeaturesd3mwrapper/RffeaturesD3MWrapper/python_pipeline_generator_rffeatures.py"
match="step_3.add_output('produce')"
insert="temporary line proportion_of_features"
sed -i "s/$match/$match\n$insert/" $file
insert="temporary line only_numeric_cols"
sed -i "s/$match/$match\n$insert/" $file

for i in "${Datasets[@]}"; do
  mkdir "../experiments_$i"
  dir="../experiments_$i/*"
  touch "../experiments_$i/dummy.meta"
  for n in $(seq 1.0 -0.1 0.75); do
    for m in "${Bools[@]}"; do

      # change HPs
      sed -i '/proportion_of_features/d' $file
      sed -i '/only_numeric_cols/d' $file
      insert="step_3.add_hyperparameter(name='proportion_of_features', argument_type=ArgumentType.VALUE,data=$n)"
      sed -i "s/$match/$match\n$insert/" $file
      insert="step_3.add_hyperparameter(name='only_numeric_cols', argument_type=ArgumentType.VALUE,data=$m)"
      sed -i "s/$match/$match\n$insert/" $file

      # generate and save pipeline + metafile
      python3 "/src/rffeaturesd3mwrapper/RffeaturesD3MWrapper/python_pipeline_generator_rffeatures.py" $i

      # test and score pipeline
      start=`date +%s`
      python3 -m d3m runtime -d /datasets/ fit-score -m *.meta -p *.json -c scores.csv
      end=`date +%s`
      runtime=$((end-start))

      if [ $runtime -lt 3600 ]; then
        echo "$i took less than 1 hour, evaluating pipeline"
        IFS=, read col1 score col3 col4 < <(tail -n1 scores.csv)
        echo "$score"
        echo "$best_score"
        if [[ $score > $best_score ]]; then
          echo "$i, $score, $n, $m, $runtime" >> ../scores.txt
          best_score=$score
          echo "$best_score"
          rm $dir
          cp *.meta "../experiments_$i/"
          cp *.json "../experiments_$i/"
        fi
      fi
      # cleanup temporary file
      rm *.meta
      rm *.json
      rm scores.csv
    done
  done

  # save best pipeline for each dataset
  cp "../experiments_$i/*.meta" ../best_pipelines/
  cp "../experiments_$i/*.json" ../best_pipelines/
  rm "../experiments_$i/*.meta"
  rm "../experiments_$i/*.json"
done