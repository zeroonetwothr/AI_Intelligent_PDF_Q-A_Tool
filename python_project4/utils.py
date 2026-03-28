from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import  ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from add import DoubaoEmbeddings
import tempfile
import os
load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("DOUBAO_MODEL"),
    api_key=os.getenv("DOUBAO_API"),
    base_url=os.getenv("DOUBAO_BASE"),
)
def qa_agent(uploaded_file, question):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_file_path = tmp.name
    try:
        loader = PyMuPDFLoader(temp_file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n", "。", "！", "？", "，", "、", ""]
        )
        texts = text_splitter.split_documents(docs)
        embeddings_model = DoubaoEmbeddings(
            model="doubao-embedding-vision-250615",
        )
        db = FAISS.from_documents(texts, embeddings_model)
        retriever = db.as_retriever()
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是一个基于给定文档内容回答问题的助手。"
                "只能使用提供的上下文回答问题，如果无法从中得到答案，请明确说明不知道。"
            ),
            ("human", "问题：{question}\n\n上下文：{context}")
        ])
        rag_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )
        result = rag_chain.invoke(question)
        return result.content
    finally:
        os.remove(temp_file_path)
