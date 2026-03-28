from langchain.embeddings.base import Embeddings
from volcenginesdkarkruntime import Ark
import os
class DoubaoEmbeddings(Embeddings):
    def __init__(self, model: str):
        self.client = Ark(api_key=os.getenv("DOUBAO_API"))
        self.model = model
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            resp = self.client.multimodal_embeddings.create(
                model=self.model,
                input=[
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            )
            embeddings.append(resp.data.embedding)
        return embeddings
    def embed_query(self, text):
        resp = self.client.multimodal_embeddings.create(
            model=self.model,
            input=[
                {
                    "type": "text",
                    "text": text
                }
            ]
        )
        return resp.data.embedding
