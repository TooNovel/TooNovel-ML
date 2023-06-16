from fastapi import FastAPI
from pydantic import BaseModel
import ModelGenerator
import Recommender
import FilterMessage
import DatabaseToCsv

app = FastAPI()

class Chat(BaseModel):
    message: str

@app.get("/recommend/{user_id}")
async def recommender(user_id: int):
    print("recommender : Received Message : " + str(user_id))
    return Recommender.run(user_id)

@app.put("/")
async def update():
    DatabaseToCsv.run()
    ModelGenerator.run()
    
@app.post("/filter")
async def messageFilter(chat: Chat):
    print("send message : " + chat.message)
    
    result = FilterMessage.run(chat.message)
    return result