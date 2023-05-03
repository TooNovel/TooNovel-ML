from fastapi import FastAPI
import ModelGenerator
import Recommender

app = FastAPI()

@app.get("/recommend/{user_id}")
async def recommender(user_id: int):
    print("recommender : Received Message : " + str(user_id))
    return Recommender.run(user_id)

@app.put("/")
async def update():
    ModelGenerator.run()