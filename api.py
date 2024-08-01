import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.requests import Request

from modules.manager import Manager

db_path = os.getenv("DB_PATH", "./database/")
db_config = os.getenv("DB_CONFIG", "./database/config.json")

manager = Manager(db_path, db_config)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/list_databases")
def list_databases():
    return JSONResponse(status_code=200, content=manager.list_databases())

@app.post("/create_database")
def create_database(db_name: str, dimension: int):
    if db_name in manager.list_databases():
        return JSONResponse(status_code=400, content={"error": "Database already exists."})
    manager.create_database(db_name, dimension)
    return JSONResponse(status_code=201, content={"message": "Database created."})

@app.post("/delete_database")
def delete_database(db_name: str):
    if db_name not in manager.list_databases():
        return JSONResponse(status_code=404, content={"error": "Database not found."})
    manager.delete_database(db_name)
    return JSONResponse(status_code=200, content={"message": "Database deleted."})

@app.post("/save_databases")
def save_databases():
    manager.save_databases()
    return JSONResponse(status_code=200, content={"message": "Databases saved."})

@app.post("/add")
async def add(db_name: str, request: Request):
    # ticketId and the vector will be in a json in the request body
    data = await request.json()
    ticketId = data.get("ticketId")
    vector = data.get("vector")

    if db_name not in manager.list_databases():
        return JSONResponse(status_code=404, content={"error": "Database not found."})
    if ticketId is None and vector is None:
        return JSONResponse(status_code=400, content={"error": "Name or vector should be provided."})
    result = manager.add(db_name, ticketId, vector)
    if result != True:
        return JSONResponse(status_code=400, content={"error": result})   
    return JSONResponse(status_code=201, content={"message": "Vector added."})


@app.post("/remove")
async def remove(db_name: str, request: Request):
    data = await request.json()
    ticketId = data.get("ticketId") if data.get("ticketId") else None
    vector = data.get("vector") if data.get("vector") else None
    if ticketId is None and vector is None:
        return JSONResponse(status_code=400, content={"error": "Name or vector should be provided."})
    if db_name not in manager.list_databases():
        return JSONResponse(status_code=404, content={"error": "Database not found."})
    result = manager.remove(db_name, ticketId, vector)
    if not result:
        return JSONResponse(status_code=404, content={"error": "Vector not found."})
    return JSONResponse(status_code=200, content={"message": "Vector removed."})

@app.post("/search")
async def search(db_name: str, request: Request):
    data = await request.json()
    query_vector = data.get("vector") if data.get("vector") else None
    k = data.get("k") if data.get("k") else 5

    if query_vector is None:
        return JSONResponse(status_code=400, content={"error": "Vector should be provided."})
    if db_name not in manager.list_databases():
        return JSONResponse(status_code=404, content={"error": "Database not found."})
    return JSONResponse(status_code=200, content=manager.search(db_name, query_vector, k))

@app.post("/search_text")
async def search_text(db_name: str, request: Request):
    data = await request.json()
    query_text = data.get("text") if data.get("text") else None
    k = data.get("k") if data.get("k") else 5

    if query_text is None:
        return JSONResponse(status_code=400, content={"error": "Vector should be provided."})
    if db_name not in manager.list_databases():
        return JSONResponse(status_code=404, content={"error": "Database not found."})
    return JSONResponse(status_code=200, content=manager.search_text(db_name, query_text, k))