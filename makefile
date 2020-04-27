build:
	@echo "Building Kung Fu D3M Primitives Image"
	DOCKER_BUILDKIT=1 docker build -t d3m-primitives . 

run:
	@echo "Running Kung Fu D3M Primitives Image"
	docker-compose run d3m-primitives