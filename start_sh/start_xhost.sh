#!/bin/bash
xhost + 
sudo chmod -R 777 /mnt/image
sudo nvpmodel -m 0
sudo jetson_clocks 
sudo chmod  666  /dev/ttyTHS0
exit 0
