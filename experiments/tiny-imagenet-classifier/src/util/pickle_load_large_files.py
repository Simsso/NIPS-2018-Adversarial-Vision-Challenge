# from here: https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
import pickle
import os.path

MAX_BYTES = 2**31 - 1

def load_large_file(file_path):
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, MAX_BYTES):
            bytes_in += f_in.read(MAX_BYTES)
    data = pickle.loads(bytes_in)
    return data