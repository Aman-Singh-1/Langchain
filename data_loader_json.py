from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import langchain
import pinecone 
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
import os
import json
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


class DocumentProcessor:
    def __init__(self, openai_api_key, pinecone_api_key, index_name):
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        os.environ["PINECONE_API_KEY"] = pinecone_api_key
        openai.api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    
    def load_data(self, file_path):
        loader = JSONLoader(file_path=file_path, jq_schema='.', text_content=False)
        data = loader.load()
        return data

    def chunk_data(self, data, chunk_size=1024, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = text_splitter.split_documents(data)
        return documents

    def setup_vector_store(self, documents):
        index = Pinecone.from_documents(documents, self.embeddings, index_name=self.index_name)
        vectorstore = PineconeVectorStore(index_name=self.index_name, embedding=self.embeddings)
        # index.delete(delete_all=True) 

        return vectorstore

    def add_documents_to_vector_store(self, vectorstore, documents, namespace=''):
        vectorstore.add_documents(documents, namespace=namespace)

# Example usage
if __name__ == "__main__":
    OPENAI_API_KEY = "sk-proj-rtDLB5iazT2RgxYIxpkoT3BlbkFJdO7Rj0uvlZYSqN9OmT7h"
    PINECONE_API_KEY = '7eca74e1-da4e-4465-b2c0-8a4a0e9945bf'
    INDEX_NAME = 'podhealth'

    processor = DocumentProcessor(openai_api_key=OPENAI_API_KEY, pinecone_api_key=PINECONE_API_KEY, index_name=INDEX_NAME)
    base_path = '/home/aman/podhealth/Coding_part/Data_loader/DataSet_MOM_GPT/Athena/Athena_dataset/synthetic_data_'
    for i in range(1, 101):
        file_path = f"{base_path}{i:03d}.json"

    # Load and process data
    # file_path = '/home/aman/podhealth/Coding_part/Data_loader/DataSet_MOM_GPT/Athena/Athena_dataset/synthetic_data_001.json'
        data = processor.load_data(file_path=file_path)
        documents = processor.chunk_data(data=data)
    
    # Setup vector store
        vectorstore = processor.setup_vector_store(documents)

    # Add documents to vector store
        processor.add_documents_to_vector_store(vectorstore, documents)
        print(f"Processed {file_path}")


