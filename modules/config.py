import os
import json

class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load_config()
        self.databases = self.config["databases"]
        self.last_sync_date = self.config.get("last_sync_date", None)

    def load_config(self):
        with open(self.config_file, "r") as f:
            self.config = json.load(f)

    def save_config(self):
        self.config["last_sync_date"] = self.last_sync_date
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=4)

    def add_database(self, db_name):
        self.databases[db_name] = db_name
        self.config["databases"] = self.databases
        self.save_config()

    def remove_database(self, db_name):
        del self.databases[db_name]
        self.config["databases"] = self.databases
        self.save_config()