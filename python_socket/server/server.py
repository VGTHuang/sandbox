#!/usr/bin/env python

# https://blog.csdn.net/sirobot/article/details/121831235

import numpy as np
import socket
import struct

import threading

import json


def action():

    server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', 80))
    server.listen(3)
    while True:
        print('start.......')
        sock, addr = server.accept()
        d = sock.recv(struct.calcsize('l'))
        total_size = struct.unpack('l',d)
        num  = total_size[0]//1024

        data = b''
        for i in range(num):
            data += sock.recv(1024)
        data += sock.recv(total_size[0]%1024)
        data = json.loads(data)

        np_array = np.array(data['array'])

        print(np_array.shape)

        sock.send(b'msg received')
        # np_array = np.frombuffer(data, dtype=np.uint8)

        sock.close()
    # sock.close()

if __name__ == '__main__':

    thread = threading.Thread(target=action)

    thread.start()
