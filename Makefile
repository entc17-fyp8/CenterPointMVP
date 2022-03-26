#########################################################
#  BUILD  #
#########################################################
# Base image
BASE_IMAGE_NAME = centerpointmvp_base
# Main image
IMAGE_NAME = centerpointmvp_ros


# Build only the base image - centerpoint_base
.PHONY: build-base
build-base: 
	docker build -f ./docker/dockerfile.centerpoint_base -t ${BASE_IMAGE_NAME} .

# Build only the main image - centerpoint_ros
.PHONY: build
build: 
	docker build -f ./docker/dockerfile.centerpoint_ros -t ${IMAGE_NAME} . 

# Build both base and main images
.PHONY: build-all
build-all: build-base
	docker build -f ./docker/dockerfile.centerpoint_ros -t ${IMAGE_NAME} . 



#########################################################
#  RUN  #
#########################################################
DOCKER_VOLUMES = \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--volume="/mnt/sda2/Batch17/Datasets/nuscenes_extracted/trainval_test:/workspace/CenterPoint/data/nuScenes":rw \
	--volume="${PWD}/CenterPoint":"/workspace/CenterPoint":rw \
	--volume="${PWD}/Checkpoints":"/workspace/Checkpoints":rw 
DOCKER_ENV_VARS = \
	--env="NVIDIA_DRIVER_CAPABILITIES=all" \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1"
DOCKER_ARGS = ${DOCKER_VOLUMES} ${DOCKER_ENV_VARS}

CONTAINER_NAME=${IMAGE_NAME}

# Run the Docker container
.PHONY: run
run:
	@docker run --name=${CONTAINER_NAME} -it -d --net=host --gpus all --ipc=host \
		${DOCKER_ARGS} ${IMAGE_NAME}

.PHONY: runit
runit:
	@docker run --name=${CONTAINER_NAME} -it --net=host --gpus all --ipc=host \
		${DOCKER_ARGS} ${IMAGE_NAME} bash

# Command to make GUI working (xhost +)
.PHONY: xhost
xhost:
	@xhost +

# Start a terminal in existing  Docker container
.PHONY: terminal
terminal:
	@docker exec -it ${CONTAINER_NAME} bash

# Stop container
.PHONY: stop
stop:
	@docker stop ${CONTAINER_NAME}
# Start a stopped container
.PHONY: start
start:
	@docker start ${CONTAINER_NAME}

# Start a stopped container
.PHONY: clean
clean:
	@docker rm ${CONTAINER_NAME}
