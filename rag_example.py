from db_utils.db_func import *
import pprint as pp
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
#from gusef import llm
from giga_utils.generate_answer import get_answer_docs

#create_db()
vector_db = get_db()

query = 'Может ли заседание закупочной комиссии быть проведено в заочной форме'

vector_store = get_db_response(query, vector_db)

pp.pprint(vector_store)

res = get_answer_docs(query, vector_store)
print(res)






