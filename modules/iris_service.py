from modules.vectordb import VectorDB
from modules.config import Config
from modules.model import EmbeddingModel
from modules.manager import Manager

from modules.irisdb_model import get_tickets
from modules.tools import clean_text, formatter

from datasets import Dataset
from torch.utils.data import DataLoader

import time
import tqdm

class IRIS_Service:
    def __init__(self, manager: Manager):
        self.batch_size = 1000
        self.manager = manager

        if "iris_default" not in self.manager.databases:
            self.manager.create_database("iris_default", self.manager.dimension)

        if "iris_task_focus" not in self.manager.databases:
            self.manager.create_database("iris_task_focus", self.manager.dimension)

    def sync_new_tickets(self):
        print("Syncing new tickets from IRIS database...")
        last_sync_date = self.manager.config.last_sync_date
        new_ticket_count = 0
        tickets = get_tickets(last_sync_date)
        tickets = list(tickets)

        print("Syncing", len(tickets), "new tickets.")

        default_sentences = []
        task_focus_sentences = []

        for t in tickets:
            t["description"] = clean_text(t["description"])
            default_sentences.append(formatter(t, "default"))
            task_focus_sentences.append(formatter(t, "task_focus"))

        # create two datasets from the sentences
        default_dataset = Dataset.from_dict({"text": default_sentences, "ticket_id": [t["ticket_id"] for t in tickets]})
        task_focus_dataset = Dataset.from_dict({"text": task_focus_sentences, "ticket_id": [t["ticket_id"] for t in tickets]})

        # create two dataloaders from the datasets
        default_dataloader = DataLoader(default_dataset, batch_size=self.manager.model.batch_size, num_workers=4, shuffle=False)
        task_focus_dataloader = DataLoader(task_focus_dataset, batch_size=self.manager.model.batch_size, num_workers=4, shuffle=False)

        # get the embeddings for the default sentences
        for batch in tqdm.tqdm(default_dataloader):
            inputs = self.manager.model.tokenize_text(batch)
            vectors = self.manager.model.get_cls_embeddings_from_inputs(inputs).tolist()
            for i in range(len(batch["ticket_id"])):
                self.manager.add("iris_default", batch["ticket_id"][i], vectors[i])

        self.manager.save_databases()

        # get the embeddings for the task focus sentences
        for batch in tqdm.tqdm(task_focus_dataloader):
            inputs = self.manager.model.tokenize_text(batch)
            vectors = self.manager.model.get_cls_embeddings_from_inputs(inputs).tolist()            
            for i in range(len(batch["ticket_id"])):
                result = self.manager.add("iris_task_focus", batch["ticket_id"][i], vectors[i])
                
                if not result:
                    print("Error adding", batch["ticket_id"][i], "to iris_task_focus database:", result)
                else:
                    new_ticket_count += 1

        self.manager.save_databases()            

        print("Synced", new_ticket_count, "new tickets.")
        self.manager.config.last_sync_date = time.strftime("%Y-%m-%d %H:%M:%S")
        self.manager.config.save_config()

        # for i in tqdm.tqdm(range(0, len(tickets), self.batch_size)):
        #     batch = tickets[i:i+self.batch_size]
        #     batch = [t for t in batch if not t["ticket_id"] in self.manager.databases["iris_default"].name]
            
        #     if len(batch) == 0:
        #         continue

        #     default_sentences = []
        #     task_focus_sentences = []
        #     for t in batch:
        #         t["description"] = clean_text(t["description"])
        #         default_sentences.append(formatter(t, "default"))
        #         task_focus_sentences.append(formatter(t, "task_focus"))

        #     vectors = self.manager.model.get_cls_embeddings(default_sentences).tolist()
        #     vectors = self.manager.model.get_cls_embeddings(task_focus_sentences).tolist()

        #     for i, t in enumerate(batch):
        #         self.manager.add("iris_default", t["ticket_id"], vectors[i])
        #         self.manager.add("iris_task_focus", t["ticket_id"], vectors[i])

        #     new_ticket_count += len(batch)

        #     self.manager.save_databases()

        return new_ticket_count