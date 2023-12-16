import time

def measure_time(func):
    def time_wrapper():
        start = time.time()
        res = func()
        end = time.time()
        delta = start - end
        print('Время работы функции --{0}-- : {1}'.format( func.__name__, delta ) )
        return res
    return time_wrapper()

