from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
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
import openai
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI  
from langchain.chains import RetrievalQA  
from pinecone import Pinecone, PodSpec
import os




PINECONE_API_KEY='7eca74e1-da4e-4465-b2c0-8a4a0e9945bf'
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
openai.api_key = "sk-proj-rtDLB5iazT2RgxYIxpkoT3BlbkFJdO7Rj0uvlZYSqN9OmT7h"
OPENAI_API_KEY="sk-proj-rtDLB5iazT2RgxYIxpkoT3BlbkFJdO7Rj0uvlZYSqN9OmT7h"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
index_name='podhealth'
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model_name = 'text-embedding-3-small'  
embeddings = OpenAIEmbeddings(  
    model=model_name,  
    openai_api_key=OPENAI_API_KEY
)

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

loader = WebBaseLoader(["https://www.autismspeaks.org/signs-autism","https://www.autismspeaks.org/what-causes-autism","https://www.autismspeaks.org/asperger-syndrome","https://www.autismspeaks.org/autism-statistics-asd","https://www.autismspeaks.org/screen-your-child","https://www.autismspeaks.org/first-concern-action","https://www.autismspeaks.org/autism-diagnostic-criteria-dsm-5","https://www.autismspeaks.org/new-autism-diagnosis","https://www.autismspeaks.org/medical-conditions-associated-autism","https://www.autismspeaks.org/sensory-issues","https://www.autismspeaks.org/access-services","https://www.autismspeaks.org/health-insurance","https://www.autismspeaks.org/caregiver-skills-training-program","https://autismsociety.org/national-programs/","https://autismsociety.org/road-to-acceptance/","https://autismsociety.org/ways-to-give/","https://autismsociety.org/autism-justice-center/","https://autismsociety.org/resources/election/","https://www.votervoice.net/AutismSociety/home","https://autismsociety.org/the-autism-experience/","https://autismsociety.org/screening-diagnosis/","https://www.disabilityscoop.com/2024/06/28/fossil-suggests-neanderthals-cared-for-child-with-down-syndrome/30943/","https://www.disabilityscoop.com/2024/06/27/social-security-ssi-increase-may-disappoint/30941/","https://www.ahany.org/","https://autismhwy.com/blog/the-forest-awaits-you/","https://pottygenius.com/blogs/blog/potty-training-a-child-with-autism-using-aba","https://asatonline.org/research-treatment/research-synopses/immersive-virtual-reality-safety-training/","https://pottygenius.com/blogs/blog/potty-training-a-child-with-autism-using-aba","https://asatonline.org/research-treatment/clinical-corner/right-data-collection/"])
docs = loader.load()
# loader = WebBaseLoader(["",])
chunk_size=1024
chunk_overlap=50
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
documents = text_splitter.split_documents(docs)
vectorstore.add_documents(documents)

