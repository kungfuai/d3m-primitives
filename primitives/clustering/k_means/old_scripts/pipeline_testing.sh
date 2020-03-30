#!/bin/bash -e 

# make directory to save functional pipelines
# Primitives=('d3m.primitives.clustering.k_means.Sloth' 'd3m.primitives.clustering.hdbscan.Hdbscan' 'd3m.primitives.dimensionality_reduction.t_distributed_stochastic_neighbor_embedding.Tsne' 'd3m.primitives.clustering.spectral_graph_clustering.SpectralClustering')
# for i in "${Primitives[@]}"; do
#   mkdir $i
#   cd $i
#   mkdir 1.0.5
#   cd 1.0.5
#   mkdir pipelines
#   mkdir pipeline_runs
#   python3 -m d3m index describe -i 4 $i > primitive.json
#   cd ../../
# done 

# Spectral Clustering Pipelines
# cd d3m.primitives.clustering.spectral_graph_clustering.SpectralClustering/1.0.5/pipelines
# # python3 /wrap/pipelines/SpectralClustering_pipeline_winter_2020.py
# cd ../pipeline_runs
# Datasets=('SEMI_1044_eye_movements_MIN_METADATA')
# for i in "${Datasets[@]}"; do

#   start=`date +%s`
#   python3 -m d3m runtime -d /datasets/ fit-score -p ../pipelines/*.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -O ${i}.yml -a /datasets/seed_datasets_current/$i/SCORE/dataset_SCORE/datasetDoc.json
#   end=`date +%s`
#   runtime=$((end-start))

#   echo "----------$i took $runtime----------"
# done


# Tsne Pipelines
# cd d3m.primitives.dimensionality_reduction.t_distributed_stochastic_neighbor_embedding.Tsne/1.0.5/pipelines
# # rm *.json
# # python3 /wrap/pipelines/winter_2020/Tsne_pipeline.py
# cd ../pipeline_runs
# Datasets=('SEMI_1044_eye_movements_MIN_METADATA')
# for i in "${Datasets[@]}"; do

#   start=`date +%s`
#   python3 -m d3m runtime -d /datasets/ fit-score -p ../pipelines/*.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -O ${i}.yml -a /datasets/seed_datasets_current/$i/SCORE/dataset_SCORE/datasetDoc.json
#   end=`date +%s`
#   runtime=$((end-start))

#   echo "----------$i took $runtime----------"
# done

# Hdbscan Pipelines
cd d3m.primitives.clustering.hdbscan.Hdbscan/1.0.5/pipelines
# python3 /wrap/pipelines/Hdbscan_pipeline_best.py
cd ../pipeline_runs
Datasets=('SEMI_1044_eye_movements_MIN_METADATA')
for i in "${Datasets[@]}"; do

  start=`date +%s`
  python3 -m d3m runtime -d /datasets/ fit-score -p ../pipelines/*.json -i /datasets/seed_datasets_current/$i/TRAIN/dataset_TRAIN/datasetDoc.json -t /datasets/seed_datasets_current/$i/TEST/dataset_TEST/datasetDoc.json -r /datasets/seed_datasets_current/$i/${i}_problem/problemDoc.json -O ${i}.yml -a /datasets/seed_datasets_current/$i/SCORE/dataset_SCORE/datasetDoc.json
  end=`date +%s`
  runtime=$((end-start))

  echo "----------$i took $runtime----------"
done

# Sloth Pipelines
# cd d3m.primitives.clustering.k_means.Sloth/1.0.5/pipelines
# python3 /wrap/pipelines/winter_2020/Sloth_pipeline.py

