import json
import os
import sys
import boto3
import streamlit as st

# Using Titan Embedding Model to generate Embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock
#from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate

# Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

# Vector Embedding and Vector Store
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA

# Bedrock Client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='ap-south-1'  
)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    docs = [doc for doc in docs if hasattr(doc, 'page_content') and doc.page_content.strip() != ""]
    return docs

# Vector embedding and vector store
def get_vector_store(docs):
    if not docs:
        st.error("No documents to embed. Please check your data folder.")
        return

    try:
        vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        vectorstore_faiss.save_local("faiss_index")
        st.success("Vector Store Created Successfully!")
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")

# def get_claude_llm():
#     # create the Anthrop Model
#     llm=Bedrock(model_id="amazon.nova-lite-v1:0", client=bedrock )
#     return llm

# def get_nova_llm():
#     def nova_invoke(prompt):
#         body = {
#             "prompt": prompt,
#             "max_gen_len": 1000,
#             "temperature": 0.5,
#             "top_p": 0.9
#         }

#         response = bedrock.invoke_model(
#             modelId="amazon.nova-lite-v1:0",
#             contentType="application/json",
#             accept="application/json",
#             body=json.dumps(body)
#         )

#         response_body = json.loads(response['body'].read())
#         return response_body['generation']

#     return nova_invoke

        
        # response = bedrock.invoke_model(
        #     modelId="amazon.nova-lite-v1:0",
        #     contentType="application/json",
        #     accept="application/json",
        #     body=json.dumps(body)
        # )
        

# def get_llama2_llm():
#     # create the Anthrop Model
#     llm=Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock )
#     return llm

def get_llama2_llm():
    def llama_invoke(prompt):
        body = {
            "prompt": prompt,
            "max_gen_len": 512,
            "temperature": 0.5,
            "top_p": 0.9
        }

        response = bedrock.invoke_model(
            modelId="meta.llama3-70b-instruct-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )

        response_body = json.loads(response['body'].read())
        return response_body['generation']

    return llama_invoke


promt_template = """ 
    Human: Use the following pieces of context to provide a concise answer to the 
    question at the end but use atleast summarize with 250 words with detailed explenations. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Assistant:
"""

PROMPT = PromptTemplate(
    template=promt_template,
    input_variables=["context", "question"]
)

# def get_response_llm(llm, vectorstore_faiss, query):
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectorstore_faiss.as_retriever(
#             search_type="similarity",
#             search_kwargs={"k": 1}
#         ),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": PROMPT}
#     )
    
#     answer = qa({"query": query})
#     return answer["result"]

def get_response_llm(llm_func, vectorstore_faiss, query):
    retriever = vectorstore_faiss.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}
    )
    relevant_docs = retriever.get_relevant_documents(query)
    
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = PROMPT.format(context=context, question=query)
    response = llm_func(prompt)
    return response


def main():
    st.set_page_config(page_title="BedrockReader", page_icon=":page_facing_up:", layout="wide")
    st.header("BedrockReader")
    
    user_question = st.text_input("Ask a question about the document:") 
    
    with st.sidebar:
        st.title("Update or Create Vector Store:")
        
        if st.button("Upadate Vectors"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector Store Updated Successfully!")
                
    # if st.button("Claude Output"):
    #     with st.spinner("Processing..."):
    #         faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
    #         llm= get_claude_llm()
            
    #         st.write(get_response_llm(llm, faiss_index, user_question))
    #         st.success("Claude Output Generated!")
    
    # if st.button("Nova Output"):
    #     with st.spinner("Processing..."):
    #         faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    #         llm_func = get_nova_llm()
    #         st.write(get_response_llm(llm_func, faiss_index, user_question))
    #         st.success("Nova Output Generated!")

            
    if st.button("Llama Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm_func = get_llama2_llm()
            st.write(get_response_llm(llm_func, faiss_index, user_question))
            st.success("Llama Output Generated!")

if __name__ == "__main__":
    main()
