import json
import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone
from pinecone.exceptions import PineconeException

class PDFProcessor:
    def __init__(self, pdf_path, pinecone_api_key, openai_api_key, index_name, model_name='text-embedding-ada-002'):
        self.pdf_path = pdf_path
        self.pinecone_api_key = pinecone_api_key
        self.openai_api_key = openai_api_key
        self.index_name = index_name
        self.model_name = model_name
        os.environ["PINECONE_API_KEY"] = pinecone_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        Pinecone.init(api_key=self.pinecone_api_key)
        self.embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)
        self.vectorstore = PineconeVectorStore(index_name=index_name, embedding=self.embeddings)

    def load_pdf(self):
        loader = UnstructuredPDFLoader(
            self.pdf_path,
            strategy='hi_res',
            extract_images_in_pdf=False,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000, # max size of chunks
            new_after_n_chars=3800, # preferred size of chunks
            combine_text_under_n_chars=2000, # smaller chunks < 2000 chars will be combined into a larger chunk
            mode='elements'
        )
        print("Loading PDF...")
        documents = loader.load()
        print("PDF loaded successfully.")
        return documents

    @staticmethod
    def stringify_metadata(documents):
        def stringify_metadata(metadata):
            return {k: json.dumps(v) if not isinstance(v, (str, int, float, bool, list)) else v for k, v in metadata.items()}

        for doc in documents:
            doc.metadata = stringify_metadata(doc.metadata)
        print("Metadata converted to strings.")
        return documents

    def process_and_upload_documents(self):
        documents = self.load_pdf()
        documents = self.stringify_metadata(documents)
        try:
            self.vectorstore.add_documents(documents)
            print("Documents successfully uploaded to Pinecone.")
        except PineconeException as e:
            print(f"Error adding documents to Pinecone: {e}")

# Usage
if __name__ == "__main__":
    pdf_path = '/home/aman/podhealth/Coding_part/Data_loader/DataSet_MOM_GPT/PDF/brainsci-10-00163.pdf'
    pinecone_api_key = '7eca74e1-da4e-4465-b2c0-8a4a0e9945bf'
    openai_api_key = 'sk-proj-rtDLB5iazT2RgxYIxpkoT3BlbkFJdO7Rj0uvlZYSqN9OmT7h'
    index_name = 'pubmed'
    
    processor = PDFProcessor(pdf_path, pinecone_api_key, openai_api_key, index_name)
    processor.process_and_upload_documents()
