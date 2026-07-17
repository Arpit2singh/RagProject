
from queue_config import queue 
from fastapi import FastAPI, UploadFile, Body , File
from fastapi.middleware.cors import CORSMiddleware
from typing import Union, List
from main import extract_text_pdf , chunk_text , embedding_create , store_on_cloud , generate_answer , process_pdf_job , process_query_job

app = FastAPI() 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def server():
    return "server is running" 

@app.post("/upload")
async def uploadpdf(pdfs : List[UploadFile] = File(...)):
   jobs_ids=[] 
   for pdf in pdfs:
       file_byte = await pdf.read()
       job = queue.enqueue(process_pdf_job , file_byte , pdf.filename )
       jobs_ids.append({"filename": pdf.filename, "job_id": job.id})
        
   
   return {"status": "success", "message": "PDF uploaded and stored successfully" , "jobs" : jobs_ids}

@app.post("/getanswer")
def getanswer(ques:str):
        
    job = queue.enqueue(process_query_job , ques)
    return {"job_id" : job.id}

@app.get("/get_result")
def get_result(job_id : str):
  
    try:
        job = queue.fetch_job(job_id)
        result  = job.return_value()
        return {"result" : result }
    except Exception as e:
        return {"error":str(e)}