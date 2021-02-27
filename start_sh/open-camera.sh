docker run -it -d --restart=always --runtime nvidia --network chepai --cpus 2 --privileged -v /etc/localtime:/etc/localtime:ro -v /data/Scanner/open-camera:/work -v /data/image:/data/image -v /usr/local/cuda:/usr/local/cuda -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --name opencv-camera open-camera:0.1

