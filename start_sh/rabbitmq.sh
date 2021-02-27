docker run -it -d  -v /etc/localtime:/etc/localtime:ro --network chepai -p 5672:5672 -p 15672:15672 --name  vbs-rabbitmq rabbitmq-server:0.2  rabbitmq-server status
