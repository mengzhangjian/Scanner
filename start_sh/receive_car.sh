docker run -it -d  --runtime nvidia --restart=always -e PYTHONIOENCODING=utf-8 -v /etc/localtime:/etc/localtime:ro -v /data/image:/data/image -v /data/Scanner/:/work -v /usr/local/cuda/:/usr/local/cuda --network chepai --name receive-car receive-car:1.1

