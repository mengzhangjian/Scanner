docker run -it -d  --runtime nvidia --restart=always -e PYTHONIOENCODING=utf-8 -v /etc/localtime:/etc/localtime:ro -v /mnt/image:/mnt/image -v /home/nvidia/Documents/tx2_chepai/:/work -v /usr/local/cuda/:/usr/local/cuda --network chepai --name receive-car-new receivce-car:0.4

