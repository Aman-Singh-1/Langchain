
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema import StrOutputParser
from operator import itemgetter
import chainlit as cl
from chainlit.input_widget import TextInput
import os
from openai import AsyncOpenAI
import chainlit as cl
import openai
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain  
from langchain_core.callbacks.base import BaseCallbackHandler
import pandas as pd






PINECONE_API_KEY='7eca74e1-da4e-4465-b2c0-8a4a0e9945bf'
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
openai.api_key = "sk-proj-rtDLB5iazT2RgxYIxpkoT3BlbkFJdO7Rj0uvlZYSqN9OmT7h"
OPENAI_API_KEY="sk-proj-rtDLB5iazT2RgxYIxpkoT3BlbkFJdO7Rj0uvlZYSqN9OmT7h"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
index_name='pubmed'
model_name = 'text-embedding-3-small'  

embeddings = OpenAIEmbeddings(  
    model=model_name,  
    openai_api_key=OPENAI_API_KEY  
)
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)


# Function to format documents
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# Function to handle chat start
@cl.on_chat_start
async def when_chat_starts():
    # Initialize ChatGPT
    chatgpt = ChatOpenAI(model_name='gpt-4o', temperature=0.1, streaming=True)

    # Create retriever and chain
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    # # System prompt
    # SYS_PROMPT = """
    #     Use only the following pieces of {context} and {history} to answer the input queries Use general knowledge if question related to Autism used resource and history
    #     Keep the answer from chat history as  .
                                   
            
       
    # """
    SYS_PROMPT="""
    Give the Query of anser in the Given format of anser dont create the extra anything
      Query: What nutritional interventions are discussed in the context of autism spectrum disorder in children?

Answer: The article discusses the benefits of a gluten-free and casein-free diet, omega-3 fatty acid supplementation, and addressing nutrient deficiencies to potentially improve symptoms in children with autism.

PMID: 28512345

Title: Nutritional Interventions and Autism Spectrum Disorder in Children

Authors: Sarah Wilson, Robert Thomas, Linda Martinez

Publication Date: 2018-03-22

Journal: Journal of Child Psychology and Psychiatry

DOI: 10.1111/jcpp.12945

PMCID: PMC5976543

PMCID URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5976543/

DOI URL: https://doi.org/10.1111/jcpp.12945


Interpreted Query: AUTISM TREATMENT BASED ON B6 RANK BY YEAR, LATEST FIRST
PMID: 29562612
Title: Comprehensive Nutritional and Dietary Intervention for Autism Spectrum Disorder-A Randomized, Controlled 12-Month Trial.
Authors: Adams JB, Audhya T, Geis E, Gehn E, Fimbres V, Pollard EL, Mitchell J, Ingram J, Hellmers R, Laake D, Matthews JS, Li K, Naviaux JC, Naviaux RK, Adams RL, Coleman DM, Quig DW
Source: Nutrients
Publication Date: 2018 Mar 17
Journal: Nutrients
DOI: 10.3390/nu10030369
PMCID: pmc-id: PMC5872787;
PMCID URL: https://www.ncbi.nlm.nih.gov/pmc/articles/pmc-id: PMC5872787;/
DOI URL: https://doi.org/10.3390/nu10030369


PMID: 35818085
Title: Evidence based recommendations for an optimal prenatal supplement for women in the US: vitamins and related nutrients.
Authors: Adams JB, Kirby JK, Sorensen JC, Pollard EL, Audhya T
Source: Matern Health Neonatol Perinatol
Publication Date: 2022 Jul 11
Journal: Maternal health, neonatology and perinatology
DOI: 10.1186/s40748-022-00139-9
PMCID: pmc-id: PMC9275129;
PMCID URL: https://www.ncbi.nlm.nih.gov/pmc/articles/pmc-id: PMC9275129;/
DOI URL: https://doi.org/10.1186/s40748-022-00139-9

Query: What interventions are effective in improving social skills in children with autism?

Answer: Effective interventions include behavioral therapy, social skills training, and parent-mediated interventions, which help improve communication, social interaction, and adaptive behaviors.

PMID: 29876543

Title: Interventions for Improving Social Skills in Children with Autism

Authors: Alice Brown, David Lee, Michael Clark

Publication Date: 2019-09-10

Journal: Pediatrics

DOI: 10.1542/peds.2019-1234

PMCID: PMC6453128

PMCID URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6453128/

DOI URL: https://doi.org/10.1542/peds.2019-1234

  


       """
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYS_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # Create memory object
    memory = ConversationBufferWindowMemory(k=20, return_messages=True)

    # Define conversation chain
    conversation_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question")
            |
            retriever
            |
            format_docs,
            input=itemgetter("question"),  # user question
            history=RunnableLambda(memory.load_memory_variables)
            |
            itemgetter("history")
        )
        |
        prompt  # prompt with above user question and context
        |
        chatgpt  # above prompt is sent to the LLM for response
        |
        StrOutputParser()  # to parse the output to show on UI
    )

    # Set session variables
    cl.user_session.set("chain", conversation_chain)
    cl.user_session.set("memory", memory)

# Function to handle user message
@cl.on_message
async def on_user_message(message: cl.Message):
    # Get the chain and memory objects from the session variables
    chain = cl.user_session.get("chain")
    memory = cl.user_session.get("memory")

    # Initialize ChatGPT message
    chatgpt_message = cl.Message(content="")

    # Define callback handler
    class PostMessageHandler(BaseCallbackHandler):
        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = []

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            source_ids = []
            for d in documents:  # retrieved documents from retriever based on user query
                metadata = {
                    "source": d.metadata["source"],
                    # "page": d.metadata["page"],
                    # "content": d.page_content[:200]
                }
                idx = (metadata["source"])
                if idx not in source_ids:  # store unique source documents
                    source_ids.append(idx)
                    self.sources.append(metadata)

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            if len(self.sources):
                sources_table = pd.DataFrame(self.sources[:3]).to_markdown()
                self.msg.elements.append(
                    cl.Text(name="Sources", content=sources_table, display="inline")
                )

    # Stream the response from ChatGPT and show in real-time
    async with cl.Step(type="run", name="QA Assistant"):
        async for chunk in chain.astream(
                {"question": message.content},
                config=RunnableConfig(callbacks=[
                    cl.LangchainCallbackHandler(),
                    PostMessageHandler(chatgpt_message)
                ]),
        ):
            await chatgpt_message.stream_token(chunk)
    await chatgpt_message.send()

    # Store the current conversation in the memory object
    memory.save_context({"input": message.content},
                        {"output": chatgpt_message.content})

  
# @cl.on_chat_start
# # this function is called when the app starts for the first time



# async def when_chat_starts():

#   # Load a connection to ChatGPT LLM
#   chatgpt = ChatOpenAI(model_name='gpt-4o', temperature=0.1,
#                        streaming=True)

#   # Add a basic system prompt for LLM behavior
#   SYS_PROMPT = """
#                Act as a helpful assistant and answer questions to the best of your ability.
#                Do not make up answers.
#                """

#   # Create a prompt template for langchain to use history to answer user prompts
#   prompt = ChatPromptTemplate.from_messages(
#     [
#       ("system", SYS_PROMPT),
#       MessagesPlaceholder(variable_name="history"),
#       ("human", "{input}"),
#     ]
#   )

#   # Create a memory object to store conversation history window
#   memory = ConversationBufferWindowMemory(k=20,
#                                           return_messages=True)

#   # Create a conversation chain
#   conversation_chain = (
#     RunnablePassthrough.assign(
#       history=RunnableLambda(memory.load_memory_variables)
#       |
#       itemgetter("history")
#     )
#     |
#     prompt
#     |
#     chatgpt
#     |
#     StrOutputParser() # to parse the output to show on UI
#   )
#   # Set session variables to be accessed when user enters prompts in the app
#   cl.user_session.set("chain", conversation_chain)
#   cl.user_session.set("memory", memory)


# @cl.on_message
# # this function is called whenever the user sends a prompt message in the app
# async def on_user_message(message: cl.Message):

#   # get the chain and memory objects from the session variables
#   chain = cl.user_session.get("chain")
#   memory = cl.user_session.get("memory")

#   # this will store the response from ChatGPT LLM
#   chatgpt_message = cl.Message(content="")

#   # Stream the response from ChatGPT and show in real-time
#   async for chunk in chain.astream(
#     {"input": message.content},
#     config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
#   ):
#       await chatgpt_message.stream_token(chunk)
#   # Finish displaying the full response from ChatGPT
#   await chatgpt_message.send()
#   # Store the current conversation in the memory object
#   memory.save_context({"input": message.content},
#                       {"output": chatgpt_message.content})

# index_name='podhealth'
# # pc = Pinecone()
# model_name = 'text-embedding-3-small'  
# embeddings = OpenAIEmbeddings(  
#     model=model_name,  
#     openai_api_key=OPENAI_API_KEY
# )
# vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)


# client = AsyncOpenAI()
# @cl.on_chat_start
# # this function is called when the app starts for the first time

