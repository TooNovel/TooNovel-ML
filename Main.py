from fastapi import FastAPI
from pydantic import BaseModel
import ModelGenerator
import Recommender
import FilterMessage
import DatabaseToCsv
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import timezone
import time

app = FastAPI()

scheduler = BackgroundScheduler()
scheduler.start()

class Chat(BaseModel):
    message: str

isUpdating = False

@app.get("/recommend/{user_id}")
async def recommender(user_id: int):
    global isUpdating

    if (isUpdating):
        time.sleep(1)
        return await recommender(user_id)
    else:
        print("recommender : Received Message : " + str(user_id))
        return Recommender.run(user_id)

@app.put("/")
async def update():
    global isUpdating
    isUpdating = True
    DatabaseToCsv.run()
    ModelGenerator.run()
    isUpdating = False
    
@app.post("/filter")
async def messageFilter(chat: Chat):
    print("send message : " + chat.message)
    
    result = FilterMessage.run(chat.message)
    return result

def modelUpdateSchedule():
    global isUpdating
    print('Model Updating By Scheduler')
    isUpdating = True
    DatabaseToCsv.run()
    ModelGenerator.run()
    isUpdating = False
    print('Model Update Done')

scheduler.add_job(modelUpdateSchedule, 'cron', hour=4, minute=0, timezone=timezone('Asia/Seoul'))
