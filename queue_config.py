import os 
from redis import Redis
from dotenv import load_dotenv
from rq import Queue

load_dotenv() 
redis_url = os.getenv("REDIS_URL")

if redis_url:
    connection=Redis.from_url(redis_url)
else:
    connection=Redis(host="localhost" , port=6379)

queue=Queue(connection=connection)        