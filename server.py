from fastapi import FastAPI, UploadFile, Body , File
from typing import Union, List
from main import extract_text_pdf , chunk_text , embedding_create , store_on_cloud , generate_answer

app = FastAPI() 

@app.get("/")
def server():
    return "server is running" 

@app.post("/upload")
async def uploadpdf(pdfs : List[UploadFile] = File(...)):
   for pdf in pdfs:
       file_byte = await pdf.read() 
       text = extract_text_pdf(file_byte)
       chunks = chunk_text(text) 
       embed = embedding_create(chunks , True)
       store_on_cloud(embed, filename=pdf.filename)   
        
   
   return {"status": "success", "message": "PDF uploaded and stored successfully"}

@app.post("/getanswer")
def getanswer(ques:str):
        
    embedding = embedding_create([ques], False) 
    answer = generate_answer(ques, embedding[0]["embedding"])
    return {"answer" : answer}