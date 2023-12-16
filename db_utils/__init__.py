from sentence_transformers import SentenceTransformer
from text_utils.prepare_text import prepare_text_for_embed
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_documents(self, texts):
    res = []
    for doc in texts:
        text = prepare_text_for_embed(doc)
        emb = model.encode(text).tolist()
        res.append(emb)
    return res

def embed_query(self, text):
    t = prepare_text_for_embed(text)
    return model.encode(t).tolist()

model.embed_documents = embed_documents.__get__(model)
model.embed_query = embed_query.__get__(model)
