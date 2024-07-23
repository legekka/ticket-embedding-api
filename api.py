import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse

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
    return manager.list_databases()

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

@app.post("/add")
def add(db_name: str, ticketId: str = None, vector: list = None):
    if db_name not in manager.list_databases():
        return JSONResponse(status_code=404, content={"error": "Database not found."})
    if ticketId is None and vector is None:
        return JSONResponse(status_code=400, content={"error": "Name or vector should be provided."})
    result = manager.add(db_name, ticketId, vector)
    if result != True:
        return JSONResponse(status_code=400, content={"error": result})   
    return JSONResponse(status_code=201, content={"message": "Vector added."})


@app.post("/remove")
def remove(db_name: str, vector: list):
    if db_name not in manager.list_databases():
        return JSONResponse(status_code=404, content={"error": "Database not found."})
    result = manager.remove(db_name, vector)
    if not result:
        return JSONResponse(status_code=404, content={"error": "Vector not found."})
    return JSONResponse(status_code=200, content={"message": "Vector removed."})

@app.post("/search")
def search(db_name: str, query_vector: list, k: int):
    if db_name not in manager.list_databases():
        return JSONResponse(status_code=404, content={"error": "Database not found."})
    return manager.search(db_name, query_vector, k)