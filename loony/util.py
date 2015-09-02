import time

def time_op(fn):
    start = time.time()
    result = fn()
    end = time.time()
    return result, end - start
