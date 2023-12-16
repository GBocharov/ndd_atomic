from langchain.prompts import PromptTemplate


map_template = """Напиши краткое содержание к следующему тексту:
"{text}"
"""
map_template = PromptTemplate.from_template(map_template)
#prompt.format( text = text )


reduse_template = """ Изложи кратко текст:
"{text}"
"""
reduse_template = PromptTemplate.from_template(reduse_template)