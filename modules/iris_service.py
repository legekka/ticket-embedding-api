from modules.vectordb import VectorDB
from modules.config import Config
from modules.model import EmbeddingModel
from modules.manager import Manager

from modules.irisdb_model import get_tickets, get_ticket_times
from modules.tools import clean_text, formatter

from datasets import Dataset
from torch.utils.data import DataLoader
import torch

import time
import tqdm

class IRIS_Service:
    def __init__(self, manager: Manager):
        self.batch_size = 1000
        self.manager = manager
        self.sync_running = False

        if "iris_default" not in self.manager.databases:
            self.manager.create_database("iris_default", self.manager.dimension)

        if "iris_task_focus" not in self.manager.databases:
            self.manager.create_database("iris_task_focus", self.manager.dimension)

    def sync_new_tickets(self):
        print("Syncing new tickets from IRIS database...")
        self.sync_running = True

        last_sync_date = self.manager.config.last_sync_date
        new_ticket_count = 0
        request_time = time.strftime("%Y-%m-%d %H:%M:%S")
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
        self.manager.config.last_sync_date = request_time
        self.manager.config.save_config()

        torch.cuda.empty_cache()

        self.sync_running = False
        return new_ticket_count
    
    def predict_spent_time(self, text: str, db_name: str):
        # get relevant tickets first
        tickets_with_distance = self.manager.search_text(db_name, text, 6)
        ticket_ids = [t["name"] for t in tickets_with_distance]

        # get ticket active_spent_time and inactive_spent_time from IRIS database
        tickets = get_ticket_times(ticket_ids)

        # group tickets by ticket_id by summing the active_spent_time and inactive_spent_time
        tickets_grouped = {}
        for ticket in tickets:
            if ticket["ticket_id"] in tickets_grouped:
                tickets_grouped[ticket["ticket_id"]]["active_spent_time"] += ticket["active_spent_time"]
                tickets_grouped[ticket["ticket_id"]]["inactive_spent_time"] += ticket["inactive_spent_time"]
            else:
                tickets_grouped[ticket["ticket_id"]] = ticket

        tickets = list(tickets_grouped.values())

        # add the normalized distance to the tickets where name == ticket_id
        for ticket in tickets:
            ticket["total_spent_time"] = ticket["active_spent_time"] + ticket["inactive_spent_time"]
            for res in tickets_with_distance:
                if ticket["ticket_id"] == res["name"]:
                    ticket["distance"] = res["distance"]

        # sort the tickets by distance
        tickets = sorted(tickets, key=lambda x: x["distance"])

        try:
            result = tickets[0]["total_spent_time"]
        except:
            return 1
        
        rounded_result = round(result * 4) / 4

        return rounded_result