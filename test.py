import requests
import tqdm

from modules.irisdb_model import *
from modules.tools import clean_text, formatter

host_url = "http://localhost:8000/"
endpoint = "search_text?db_name=iris_default"

k = 5

def get_relevant_tickets(ticket_text):
    jsonbody = {
        "text": ticket_text,
        "k": k + 1
    }

    response = requests.post(host_url + endpoint, json=jsonbody)

    result = response.json()

    # convert all result.distance to float
    for res in result:
        res["distance"] = float(res["distance"])

    # remove the first result if it's 0
    if result[0]["distance"] == 0:
        result = result[1:]
    else:
        result = result[:-1]

    filtered_result = result
    return filtered_result

def calculate_predicted_time(tickets_with_distance):
    # get the tickets from the database that are in the filtered result
    ticket_ids = [res["name"] for res in tickets_with_distance]

    tickets = (
        Ticket.select(
            Ticket.id.alias("ticket_id"),
            WorkLog.public_spent_time.alias("active_spent_time"),
            WorkLog.spent_time.alias("inactive_spent_time")
        )
        .join(WorkLog, on=(Ticket.id == WorkLog.ticket_id))
        .where(Ticket.id.in_(ticket_ids))
        .dicts()
    )

    tickets = list(tickets)

    # first we need to group the tickets by ticket_id by summing the active_spent_time and inactive_spent_time
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

def get_ticket_by_id(ticket_id):
    tickets = (
        Ticket.select(
            Ticket.id.alias("ticket_id"),
            Partner.name.alias("partner"),
            Contact.name.alias("contact"),
            Ticket.subject,
            Ticket.description,
            OperationServiceLevel.name.alias("osl"),
            OperationServiceLevelType.name.alias("oslt"),
            TicketType.name.alias("type"),
            Ticket.priority,
            UserGrade.name.alias("user_grade")
        )
        .join(Contact, on=(Ticket.contact_id == Contact.id))
        .join(Partner, on=(Contact.partner_id == Partner.id))
        .join(OperationServiceLevel, on=(Ticket.operation_service_level_id == OperationServiceLevel.id))
        .join(OperationServiceLevelType, on=(OperationServiceLevel.operation_service_level_type_id == OperationServiceLevelType.id))
        .join(TicketType, on=(Ticket.ticket_type_id == TicketType.id))
        .join(User, on=(Ticket.resolver_id == User.id))
        .join(UserGrade, on=(User.user_grade_id == UserGrade.id))
        .where(Ticket.id == ticket_id)
        .dicts()
        .iterator()
    )

    tickets = list(tickets)

    try:
        ticket = tickets[0]
    except:
        return None
    
    return ticket

def get_ticket_time_by_id(ticket_id):
    ticket = (
        Ticket.select(
            Ticket.id.alias("ticket_id"),
            WorkLog.public_spent_time.alias("active_spent_time"),
            WorkLog.spent_time.alias("inactive_spent_time")
        )
        .join(WorkLog, on=(Ticket.id == WorkLog.ticket_id))
        .where(Ticket.id == ticket_id)
        .dicts()
    )

    ticket = list(ticket)

    try:
        # we need to sum all worklogs for this ticket
        total_time = 0
        for worklog in ticket:
            total_time += worklog["active_spent_time"] + worklog["inactive_spent_time"]
    except:
        return -1

    return total_time


def load_tickets_ids():
    import json
    with open("tickets.json", "r") as f:
        ticket_ids = json.load(f)

    return ticket_ids

if __name__ == "__main__":

    loop = tqdm.tqdm(load_tickets_ids())

    losses = []
    accuracy = 0
    accuracy2 = 0
    accuracy3 = 0

    for ticket_id in loop:
        ticket = get_ticket_by_id(ticket_id)

        ticket["description"] = clean_text(ticket["description"])
        ticket_text = formatter(ticket, "task_focus")

        tickets_with_distance = get_relevant_tickets(ticket_text)
        predicted_time = calculate_predicted_time(tickets_with_distance)

        original_time = get_ticket_time_by_id(ticket_id)
        if original_time == -1:
            print(original_time)
            continue

        loss = abs(predicted_time - original_time)
        losses.append(loss)

        mean_loss = sum(losses) / len(losses)
        
        if loss == 0:
            accuracy += 1
        if loss <= 0.25:
            accuracy2 += 1
        if loss <= 0.5:
            accuracy3 += 1
        avg_accuracy = accuracy / len(losses)
        avg_accuracy2 = accuracy2 / len(losses)
        avg_accuracy3 = accuracy3 / len(losses)
        loop.set_description(f"Loss: {loss:.2f} | Mean Loss: {mean_loss:.3f} | Accuracy: {avg_accuracy:.3f} | Accuracy2: {avg_accuracy2:.3f} | Accuracy3: {avg_accuracy3:.3f} | P Time: {predicted_time} | O Time: {original_time} | Ticket ID: {ticket_id}")

        
    print(f"Mean Loss: {sum(losses) / len(losses):.3f}")
    print(f"Max Loss: {max(losses):.2f}")
    print(f"Accuracy: {accuracy / len(losses):.3f}")
    print(f"Accuracy if 0.25 error allowed: {accuracy2 / len(losses):.3f}")
    print(f"Accuracy if 0.5 error allowed: {accuracy3 / len(losses):.3f}")
    print(f"Total Tickets: {len(losses)}")
