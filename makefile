build:
	@echo "Building Image"
	DOCKER_BUILDKIT=1 docker build -t yonder-primitives . 

run-a:
	@echo "Running Yonder Primitives Image"
	docker run --rm -t -i --mount type=bind,source=/Users/jgleason/Documents/NewKnowledge/D3M/D3M-Primitives/,target=/yonder-primitives --mount type=bind,source=/Users/jgleason/Documents/NewKnowledge/D3M/static_volumes/,target=/static_volumes --mount type=bind,source=/Users/jgleason/Documents/NewKnowledge/D3M/datasets,target=/datasets yonder-primitives

run:
	@echo "Running Yonder Primitives Image"
	docker-compose run yonder-primitives