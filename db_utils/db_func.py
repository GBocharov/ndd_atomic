from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
import pprint as pp
import chromadb
from db_utils import model
from text_utils.prepare_text import prepare_text_for_db
from striprtf.striprtf import rtf_to_text

def create_db(persist_directory  : str = 'vdb_langchain_doc_small', file_directory : str = 'data'):
    loader = DirectoryLoader(file_directory, glob="./*.txt")
    docs = loader.load()

    # text preapare
    for doc in docs:
        doc.page_content = prepare_text_for_db(doc.page_content)

   #embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 20)
    texts = text_splitter.split_documents(docs)

    for text in texts:
        text.page_content = text.page_content.replace(r'\n', ' ')

    persist_directory = persist_directory

    vectordb = Chroma.from_documents(documents = texts,
                                     embedding = model
                                     ,persist_directory = persist_directory
                                     )

def print_db(persist_directory  : str = 'vdb_langchain_doc_small', file_directory : str = 'data'):
    vectordb = Chroma(persist_directory = persist_directory, embedding_function = model)
    data = vectordb._collection.get(include = ['documents','metadatas','embeddings'])
    size = len(data['embeddings'])
    pp.pprint(size)


def delete_db(persist_directory  : str = '../vdb_langchain_doc_small', file_directory : str = 'data'):
    pass

def get_db(persist_directory  : str = 'vdb_langchain_doc_small'):
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=model)
    return vectordb

def get_db_response(query, db):
    docs = db.similarity_search_with_score(query)
    return docs

def add_file_to_db(persist_directory  : str = 'vdb_langchain_doc_small', file_path : str = "data/1.txt"):
    loader = TextLoader(file_path, autodetect_encoding = True)
    docs = loader.load()

    for doc in docs:
        doc.page_content = prepare_text_for_db(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 20)
    texts = text_splitter.split_documents(docs)

    for text in texts:
        text.page_content = text.page_content.replace(r'\n', ' ')

    vectordb = Chroma(persist_directory = persist_directory, embedding_function = model)
    vectordb.add_documents(docs)


def create_detailed_db(persist_directory  : str = '../vdb_langchain_doc_small', file_directory : str = '../data/постановления'):


    import codecs
    with open("../data/постановления/Постановление Правительства РФ от 03.12.2020 N 2013.rtf", 'r', encoding='cp1251', errors='ignore') as file:
        rtf_text = file.read()
    text = rtf_to_text(rtf_text)
    print(text)

    # text preapare
   #  for doc in docs:
   #      doc.page_content = prepare_text_for_db(doc.page_content)
   #
   # #embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
   #
   #  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 20)
   #  texts = text_splitter.split_documents(docs)
   #
   #  for text in texts:
   #      text.page_content = text.page_content.replace(r'\n', ' ')
   #
   #  persist_directory = persist_directory
   #
   #  vectordb = Chroma.from_documents(documents = texts,
   #                                   embedding = model
   #                                   ,persist_directory = persist_directory
   #                                   )

if __name__ == '__main__':
    create_detailed_db()