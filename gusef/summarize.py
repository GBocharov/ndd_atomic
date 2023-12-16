from collections import namedtuple
import time

from gusef import summarize_chain
import time

def map_reduce_summarize_docs(docs):
    start_time = time.time()
    res = summarize_chain.run(docs)
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)
    return res

def stuff_summarize_docs(docs):
    start_time = time.time()
    res = summarize_chain.run(docs)
    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)
    return res
