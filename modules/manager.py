import os

from modules.vectordb import VectorDB
from modules.config import Config
from modules.model import EmbeddingModel

class Manager:
    def __init__(self, db_path, config_file):
        self.db_path = db_path
        self.config = Config(config_file)
        self.model = EmbeddingModel("NYTK/PULI-BERT-Large")
        print("Model loaded.")
        self.databases = {}
        if len(self.config.databases) > 0:
            self.load_databases()
        print("Loaded", len(self.databases), "databases.")

    def load_databases(self):
        for db_name in self.config.databases.keys():
            self.databases[db_name] = VectorDB()
            self.databases[db_name].load(os.path.join(self.db_path, db_name + ".index"))
            print("Loaded", len(self.databases[db_name]), "vectors in", db_name)

    def save_databases(self):
        for db_name, db in self.databases.items():
            db.save(os.path.join(self.db_path, db_name + ".index"))

    def create_database(self, db_name, dimension):
        self.databases[db_name] = VectorDB()
        self.databases[db_name].create_database(dimension)
        self.config.add_database(db_name)
        self.save_databases()

    def add(self, db_name, name, vector):
        result = self.databases[db_name].add(name, vector)
        return result
    
    def remove(self, db_name, name=None, vector=None):
        result = self.databases[db_name].remove(name, vector)
        return result

    def search(self, db_name, query_vector, k):
        return self.databases[db_name].search(query_vector, k)
    
    def search_text(self, db_name, query_text, k):
        query_vector = self.model.get_cls_embeddings([query_text])[0]
        return self.search(db_name, query_vector, k)
    
    def delete_database(self, db_name):
        del self.databases[db_name]
        self.config.remove_database(db_name)
        os.remove(os.path.join(self.db_path, db_name + ".index"))

    def list_databases(self):
        return list(self.databases.keys())