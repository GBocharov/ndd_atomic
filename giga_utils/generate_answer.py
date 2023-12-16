from langchain.schema import HumanMessage, SystemMessage
from giga_utils import chat

def get_answer_text(request, text):
    messages = [
        SystemMessage(
            content = "Используя тексты ниже ответь на вопрос."
                      "Вопрос : {} "
                      "Тексты : {}".format(request, text)
        )
    ]
    messages.append(HumanMessage(content=text))
    res = chat(messages)
    messages.append(res)
    return res.content

def get_answer_docs(request, docs):

    context = '\n'.join(  ['Текст {} : \n {}'.format(n, d[0].page_content ) for n, d in enumerate(docs)])

    res = get_answer_text(request, context)

    return res

if __name__ == '__main__':
    ...