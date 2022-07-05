#!/usr/bin/env python

# https://blog.csdn.net/sirobot/article/details/121831235

import numpy as np
import struct
import socket
import os
import json

if __name__ == '__main__':
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('127.0.0.1', 80))
    print('connect success....')

    np_array = np.zeros((10, 10, 10)).astype(int)

    np_array_dims = np_array.shape

    send_info = {
        'array': np_array.tolist()
    }

    send_info = json.dumps(send_info).encode()

    # np_array = np_array.tobytes()

    print(len(send_info)) # 8000 bytes

    send_size_bytes = struct.pack('l', len(send_info))

    # filepath = '1.png'
    # size =  os.stat(filepath).st_size
    # f = struct.pack('l', os.stat(filepath).st_size)

    client.send(send_size_bytes)
    client.sendall(send_info)

    s = client.recv(1024)
    print(s)

    client.close()