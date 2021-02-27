#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename    :test_redis.py
@brief       :redis封装
@time        :2020/06/13 14:47:18
@author      :hscoder
@versions    :1.0
@email       :hscoder@163.com
@usage       :
'''


from redis import StrictRedis
class RfidRedis(object):
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        self.host = host
        self.port = port
        self.db = 0
        self.password = password

        self.redis = StrictRedis(self.host, self.port,
                                 db=self.db, password=self.password)

    # add element
    def add(self, key='rfid', value="000000000000"):
        self.redis.sadd(key, value)

    # get all element
    def get(self, key='rfid'):
        all_rfid = list(self.redis.smembers(key))
        all_rfid = [i.decode('utf-8') for i in all_rfid]
        sort_all_rfid = sorted(all_rfid , key = lambda item : item.split('_'))
        return sort_all_rfid

    # remove element
    def remove(self, key='rfid', value="0000000000.0000000_+116.393700,+40.939895_000000000000.jpg"):
        if self.redis.sismember(key, value):
            return self.redis.srem(key, value)
        else:
            return None

    def remove_all(self , key = 'rfid' , value = "000000000000"):
        all_rfid = self.get(key=key)   
        if key == 'rfid':  
            remove_rfid = list(filter(lambda x : x.split('_')[2][:-4] == value , all_rfid))
        else:
            remove_rfid = list(filter(lambda x : x.split('_')[1] == value , all_rfid))

        for r in remove_rfid:
            res = self.remove(key = key , value = r)

    def clear(self):
        for key in self.redis.keys():
            self.redis.delete(key)

    # get the length of redis
    def size(self, key='rfid'):
        return self.redis.scard(key)


# if __name__ == "__main__":
#     my_rfid = RfidRedis()
#     #my_rfid.add(value='6')
#     #my_rfid.add(value='2')
#     #my_rfid.add(value='3')

#     val = my_rfid.get()
#     # print(val)
#     print('---------------------------------')
#     my_rfid.remove_all(key='rfid' , rfid='E000D2000109')
#     print('---------------------------------')
#     print(my_rfid.get())



    # my_rfid.clear()
    # val = my_rfid.get()
    # print(val)
    # for i in val:
    #     print(i.decode('utf-8'))

    # print(my_rfid.size())
    # my_rfid.remove(value='5')
    # print(my_rfid.size())
