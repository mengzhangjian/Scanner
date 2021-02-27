docker run -it -d --restart=always --runtime nvidia --network chepai --cpus 2 --privileged -v /etc/localtime:/etc/localtime:ro -v /mnt/opencv-3.4.3:/opencv -v /home/nvidia/Documents/tx2_chepai/open-camera:/work -v /mnt/image:/mnt/image -v /usr/local/cuda:/usr/local/cuda -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --name opencv-dnn open-camera:0.1

