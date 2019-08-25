#!/usr/bin/env bash

# pull images from dockerhub

# images to pull
images=(
    kag_python
    kag_mlbox
    kag_h2o
    kag_tpot
)

# function to push an image to docker hub
pull_image () {

    image_to_pull=${1}
    echo "pulling image: " ${image_to_pull}

    # pull from docker hub
    docker push dsimages/${image_to_pull}

    # create tag for local use
    docker tag dsimages/${image_to_pull} ${image_to_pull}

}


#
#  pull images
#
for image in ${images[@]}
do
    pull_image ${image}
done