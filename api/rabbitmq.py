import pika
import pika.exceptions
from api.logger import Logger
import logging
import json
logger = Logger(log_file_name='./log/mq_status.txt', log_level=logging.DEBUG, logger_name="mq_test").get_log()


class RabbitMqServer:

    def __init__(self, queue):

        # self.user_id = user_id
        # self.password = password
        self.exchange = ''
        # self.exchange_type = 'topic'
        self.queue = queue
        # self.routing_key = routing_key
        # self.cred = pika.PlainCredentials(self.user_id, self.password)
        self.params = pika.ConnectionParameters('localhost',
                                                heartbeat=0)
        try:
            self.connection = pika.BlockingConnection(self.params)
        except pika.exceptions.ConnectionClosed as err:

            self.reconnect()
            raise logger.error(err)

        self.channel = self.connection.channel()

    def set_consume(self, callback):

        self.channel.exchange_declare(exchange=self.exchange, durable=True)
        self.channel.queue_declare(queue=self.queue)
        self.channel.queue_bind(exchange=self.exchange, queue=self.queue)

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(self.queue, callback, auto_ack=True)

    def start_consumer(self, callback, retry=2):

        try:
            self.channel.start_consuming()
        except pika.exceptions.AMQPError as e:
            if retry <= 0:
                raise logger.error(e)
            self.reconnect()
            self.set_consume(callback)
            self.start_consumer(callback, retry - 1)

        except pika.exceptions.ConnectionClosed as er:
            if retry <= 0:
                raise logger.error(er)
            self.reconnect()
            self.set_consume(callback)
            self.start_consumer(callback, retry - 1)

    def set_publish(self):

        self.channel.exchange_declare(exchange=self.exchange, durable=True)
        self.channel.queue_declare(queue=self.queue)
        self.channel.queue_bind(exchange=self.exchange, queue=self.queue)

    def start_publish(self, msg, retry=2):

        body = json.dumps(msg)
        try:
            self.channel.basic_publish(exchange=self.exchange, routing_key='hello',
                                       body=body)

        except pika.exceptions.ConnectionClosed as err:
            if retry <= 0:
                raise logger.error(err)

            self.reconnect()
            self.start_publish(msg, retry - 1)
        except pika.exceptions.AMQPError as e:
            if retry <= 0:
                raise logger.error(e)

            self.reconnect()
            self.start_publish(msg, retry - 1)

    def reconnect(self):
        """Reconnect to rabbitmq Server"""

        self.connection = pika.BlockingConnection(self.params)
        self.channel = self.connection.channel()

        try:
            self.channel.exchange_declare(exchange=self.exchange, durable=True)
            self.channel.queue_bind(exchange=self.exchange, queue=self.queue)
        except pika.exceptions.ChannelClosed as err:

            self.connection = pika.BlockingConnection(self.params)
            self.channel = self.connection.channel()
            raise logger.error(err)

    def acknowledge(self, method):

        if method:
            try:
                self.channel.basic_ack(delivery_tag=method.delivery_tag)
            except pika.exceptions.AMQPError as err:
                raise logger.error(err)









