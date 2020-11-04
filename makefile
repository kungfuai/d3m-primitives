build:
	@echo "Building Kung Fu D3M Primitives Image"
	DOCKER_BUILDKIT=1 docker build -t kf-d3m-primitives . 

volumes:
	@echo "Downloading large static files"
	docker-compose run --rm kf-d3m-primitives python3 download_volumes.py
run:
	@echo "Running Kung Fu D3M Primitives Image"
	docker-compose run --rm kf-d3m-primitives

run-gpu:
	@echo "Running Kung Fu GPU D3M Primitives Image"
	docker run --rm --runtime nvidia \
		--mount type=bind,source=/home/ubuntu/d3m/d3m-primitives/datasets,target=/datasets \
		--mount type=bind,source=/home/ubuntu/d3m/d3m-primitives/static_volumes,target=/static_volumes \
		--mount type=bind,source=/home/ubuntu/d3m/d3m-primitives/scratch_dir,target=/scratch_dir \
		--mount type=bind,source=/home/ubuntu/d3m/d3m-primitives/tests,target=/tests \
		--mount type=bind,source=/home/ubuntu/d3m/d3m-primitives/kf_d3m_primitives,target=/kf_d3m_primitives \
		-it kf-d3m-primitives /bin/bash

test:
	@echo "Running tests for Kung Fu D3M Primitives Image"
	docker-compose run --rm --entrypoint python3 kf-d3m-primitives -m pytest -s --rootdir=tests tests

annotations:
	@echo "Generating json annotations for all primitives"
	docker-compose run --rm kf-d3m-primitives python3 generate_annotations.py

pipelines-cpu:
	@echo "Generating pipeline run documents for all primitives"
	docker-compose run --rm kf-d3m-primitives python3 generate_pipelines.py

pipelines-gpu:
	@echo "Generating pipeline run documents for all primitives"
	docker run --rm --runtime nvidia \
		--mount type=bind,source=/home/ubuntu/d3m-primitives/annotations,target=/annotations \
		--mount type=bind,source=/home/ubuntu/d3m-primitives/datasets,target=/datasets \
		--mount type=bind,source=/home/ubuntu/d3m-primitives/static_volumes,target=/static_volumes \
		--mount type=bind,source=/home/ubuntu/d3m-primitives/scratch_dir,target=/scratch_dir \
		--mount type=bind,source=/home/ubuntu/d3m-primitives/pipeline_scores,target=/pipeline_scores \
		kf-d3m-primitives python3 generate_pipelines.py --gpu=True
