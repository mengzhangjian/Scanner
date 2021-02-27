#!/bin/bash

#source ~/.bashrc
passward='nvidia'
sudo nvpmodel -m 0
sudo jetson_clocks

path='/data/tx2_chepai/scripts/'
button='button_interrupt.py'
button_log='button.log'
monitor='monitor_disk.py'
monitor_log='monitor.log'

echo $passward | sudo python3 $path$button  > $path$button_log  2>&1 &
echo $passward | sudo python3 -u $path$monitor > $path$monitor_log  2>&1 &

