docker run -it -d  -v /etc/localtime:/etc/localtime:ro --network chepai -p 6379:6379 --name redis redis-server:0.1 redis-server --protected-mode no
