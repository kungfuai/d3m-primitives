build:
	@echo "Building Kung Fu D3M Primitives Image"
	DOCKER_BUILDKIT=1 docker build -t kf-d3m-primitives . 

volumes:
	@echo "Downloading large static files"
	docker-compose run --rm kf-d3m-primitives python3 kf-d3m-primitives/download_volumes.py
run:
	@echo "Running Kung Fu D3M Primitives Image"
	docker-compose run --rm kf-d3m-primitives

test:
	@echo "Running tests for Kung Fu D3M Primitives Image"
	docker-compose run --rm --entrypoint python3 kf-d3m-primitives -m pytest kf-d3m-primitives/tests/test_hdbscan.py

annotations:
	@echo "Generating json annotations for all primitives"
	docker-compose run --rm kf-d3m-primitives python3 kf-d3m-primitives/generate_annotations.py

pipelines:
	@echo "Generating pipeline run documents for all primitives"
	docker-compose run --rm kf-d3m-primitives python3 kf-d3m-primitives/generate_pipelines.py 