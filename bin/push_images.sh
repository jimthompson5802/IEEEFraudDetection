#!/usr/bin/env bash

# push docker images to dockerhub
# assumes already logged into dockerhub

# images to push
images=(
    kag_python
    kag_mlbox
    kag_mlboxgpu
    kag_h2o
    kag_tpot
    kag_tfgpu
)

# function to push an image to docker hub
push_image () {

    image_to_push=${1}
    echo "pushing image: " ${image_to_push}

    # create tag for docker hub
    docker tag ${image_to_push} dsimages/${image_to_push}

    # push to docker hub
    docker push dsimages/${image_to_push}

}


#
#  push images
#
for image in ${images[@]}
do
    push_image ${image}
done