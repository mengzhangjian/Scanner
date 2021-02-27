import pika
import json

"""rabbitmq initialize"""
credentials = pika.PlainCredentials('chepai', '123456')
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='vbs-rabbitmq', port=5672, credentials=credentials, heartbeat=0))
channel = connection.channel()

channel.queue_declare(queue='send')
a = {"Hello": "world"}
channel.basic_publish(exchange='', routing_key='send', body=json.dumps(a)) 


