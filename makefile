build:
	@echo "Building Image"
	DOCKER_BUILDKIT=1 docker build -t yonder-primitives . 

run:
	@echo "Running Yonder Primitives Image"
	docker-compose run yonder-primitives