from db_utils.db_func import *
import pprint as pp
#from gusef import llm
from giga_utils.generate_answer import get_answer_docs

#create_db()

def split_text(text: str, max_chars: int = 4096) -> list:
    result = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    return result

def telegram_prepare(answer):

    final = answer.\
        replace("</s>", "").replace("<s>", "").replace("</unk>", "").replace("<unk>", "").\
            replace("</n>", "").replace("<n>", "").replace("</p>", "").replace("<p>", "")
    return split_text(final)


def get_answer(req):
    vector_db = get_db()

    query = req #'Может ли заседание закупочной комиссии быть проведено в заочной форме'

    vector_store = get_db_response(query, vector_db)

    pp.pprint(vector_store)

    res = get_answer_docs(query, vector_store)

    telegram_res = telegram_prepare(res)

    return telegram_res

