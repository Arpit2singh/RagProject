from typing import List
from openai import OpenAI
import os 
from dotenv import load_dotenv
from pypdf import PdfReader
import io
import chromadb
import google.generativeai as genai 
import uuid
from fastapi import FastAPI, UploadFile, Body , File
from typing import Union, List


load_dotenv()  

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_text_pdf(pdf_path : bytes)->str:
    reader = PdfReader(io.BytesIO(pdf_path))
   
    #divide the chunks in 500 overlapping 50 words
    text = "" 
    for page in reader.pages:
        page_text = page.extract_text() 
        if page_text : 
            text+=page_text
    return text 

def chunk_text(text : str , chunk_size=500 , overlap=50):
    chunks=[]
    start=0 
    
    while start < len(text):
        end = start + chunk_size 
        chunk = text[start: end] 
        start += chunk_size - overlap
        chunks.append(chunk)
    return chunks    
            




def embedding_create(chunks:list , task_type:bool):
    if task_type ==True:
        task_type="retrieval_document"
    else:
        task_type="retrieval_query"    
    embed_chunks = [] 
    for text in chunks:
        embed = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type=task_type,
        )
        embed_chunks.append({"text":text,"embedding":embed["embedding"]}) 
    
    return embed_chunks     

def get_chroma_client():
    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE")
    if api_key and tenant and database:
        return chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database
        )
    else:
        return chromadb.PersistentClient(path="./chroma_db")

def store_on_cloud(chunks : list , filename : str):
    chroma_client = get_chroma_client()
    collection = chroma_client.get_or_create_collection("PDFS")
    
    documents=[chunk["text"] for chunk in chunks]
    embeddings=[chunk["embedding"] for chunk in chunks]
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    metadatas = [{"source": filename} for _ in chunks]
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    return collection
    
def generate_answer(question : str , embeddings : list):
   client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
   )
   chroma_client = get_chroma_client() 
   collection = chroma_client.get_or_create_collection("PDFS")
   result = collection.query(
       query_embeddings=[embeddings],
       n_results=3
   )
   context = "\n".join(result["documents"][0]) 
    
   
   system_instruction= f"""
    You are an AI assistant designed to answer questions based on the text provided. 
    Read the question carefully and then use the context (text chunks) to find the most 
    relevant information to answer the question accurately. If the information is not 
    present in the text, respond with: "I cannot answer this question based on the provided text."
    Avoid using any external knowledge.
    use this context {context}
    Question: {question}
    give the ans in structure format in points and where they are come from name and there meta data 
   """

   response = client.chat.completions.create(
       model="llama-3.3-70b-versatile",
       messages=[{ "role":"system" , "content":system_instruction},{ "role":"user" , "content":question}]
   ) 
    
   return response.choices[0].message.content  


def process_pdf_job(file_bytes:bytes , filename : str):
       text = extract_text_pdf(file_bytes)
       chunks = chunk_text(text) 
       embed = embedding_create(chunks , True)
       store_on_cloud(embed, filename=filename)   
       
       return {"status": "success", "message": "PDF uploaded and stored successfully"}
   

def process_query_job(ques : str):
    embedding = embedding_create([ques], False) 
    answer = generate_answer(ques, embedding[0]["embedding"])
    return {"answer" : answer}   
   