import sys
import time
from signal import signal, SIGINT

import threading

import numpy as np
import socket
import struct

def action():

    server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 80))
    server.listen(3)
    while True:
        print("start.......")
        sock, addr = server.accept()
        d = sock.recv(struct.calcsize("l"))
        try:
            total_size = struct.unpack("l",d)
        except:
            print('end!!!!!!!')
            break
        num  = total_size[0]//1024

        data = b''
        for i in range(num):
            data += sock.recv(1024)
        data += sock.recv(total_size[0]%1024)

        np_array = np.frombuffer(data, dtype=np.uint8)

        print(np_array)

        sock.close()
    # sock.close()


class Session:

    def __enter__(self):
        signal(SIGINT, self._sigint_handler)
        
        thread = threading.Thread(target=action)
        thread.start()

    def __exit__(self, type, value, traceback):
        print('Exiting session...')

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", 80))
        client.send(b'end')

        self._do_cleanup()

    def _do_cleanup(self):
        print('Cleaning up...')

    def _sigint_handler(self, signal_received, frame):
        print('Ctrl + C handler called')

        self.__exit__(None, None, None)
        sys.exit(0)


if __name__ == '__main__':
    with Session():
        print('Session started')