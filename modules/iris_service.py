from modules.vectordb import VectorDB
from modules.config import Config
from modules.model import EmbeddingModel
from modules.manager import Manager

from modules.irisdb_model import get_tickets
from modules.tools import clean_text, formatter

import time
import tqdm

class IRIS_Service:
    def __init__(self, manager: Manager):
        self.batch_size = 1000
        self.manager = manager

        if "iris_default" not in self.manager.databases:
            self.manager.create_database("iris_default", 1024)

        if "iris_task_focus" not in self.manager.databases:
            self.manager.create_database("iris_task_focus", 1024)

    def sync_new_tickets(self):
        last_sync_date = self.manager.config.last_sync_date
        new_ticket_count = 0
        tickets = get_tickets(last_sync_date)
        tickets = list(tickets)

        self.manager.config.last_sync_date = time.strftime("%Y-%m-%d %H:%M:%S")
        self.manager.config.save_config()

        for i in tqdm.tqdm(range(0, len(tickets), self.batch_size)):
            batch = tickets[i:i+self.batch_size]
            batch = [t for t in batch if not t["ticket_id"] in self.manager.databases["iris_default"].name]
            
            if len(batch) == 0:
                continue

            default_sentences = []
            task_focus_sentences = []
            for t in batch:
                t["description"] = clean_text(t["description"])
                default_sentences.append(formatter(t, "default"))
                task_focus_sentences.append(formatter(t, "task_focus"))

            default_vectors = self.manager.model.get_cls_embeddings(default_sentences).tolist()
            task_focus_vectors = self.manager.model.get_cls_embeddings(task_focus_sentences).tolist()

            for i, t in enumerate(batch):
                self.manager.add("iris_default", t["ticket_id"], default_vectors[i])
                self.manager.add("iris_task_focus", t["ticket_id"], task_focus_vectors[i])

            new_ticket_count += len(batch)

            self.manager.save_databases()

        return new_ticket_count