import os
from peewee import *

db_name = os.getenv("IRISDB_NAME", "irisdb")
db_user = os.getenv("IRISDB_USER", "iris")
db_password = os.getenv("IRISDB_PASSWORD", "iris")
db_host = os.getenv("IRISDB_HOST", "localhost")
db_port = os.getenv("IRISDB_PORT", "5432")

db = PostgresqlDatabase(db_name, user=db_user, password=db_password, host=db_host, port=db_port)

class BaseModel(Model):
    class Meta:
        database = db

class Ticket(BaseModel):
    id = CharField(primary_key=True, max_length=12)
    subject = CharField(max_length=250)
    description = TextField()
    priority = SmallIntegerField()
    contact_id = CharField(max_length=12)
    operation_service_level_id = CharField(max_length=12)
    ticket_type_id = CharField(max_length=12)
    resolver_id = CharField(max_length=12)
    ticket_state_id = CharField(max_length=12)
    create_date = DateTimeField()

    class Meta:
        table_name = "ticket"
        schema = "sd"

class Partner(BaseModel):
    id = CharField(primary_key=True, max_length=12)
    name = CharField(max_length=254)

    class Meta:
        table_name = "partner"
        schema = "partner"

class Contact(BaseModel):
    id = CharField(primary_key=True, max_length=12)
    name = CharField(max_length=254)
    partner_id = CharField(max_length=12)

    class Meta:
        table_name = "contact"
        schema = "partner"

class OperationServiceLevel(BaseModel):
    id = CharField(primary_key=True, max_length=12)
    name = CharField(max_length=150)
    operation_service_level_type_id = CharField(max_length=12)

    class Meta:
        table_name = "operation_service_level"
        schema = "sd"

class OperationServiceLevelType(BaseModel):
    id = CharField(primary_key=True, max_length=12)
    name = CharField(max_length=254)

    class Meta:
        table_name = "operation_service_level_type"
        schema = "sd"

class TicketType(BaseModel):
    id = CharField(primary_key=True, max_length=12)
    name = CharField(max_length=254)

    class Meta:
        table_name = "ticket_type"
        schema = "sd"

class User(BaseModel):
    id = CharField(primary_key=True, max_length=12)
    long_name = CharField(max_length=254)
    user_grade_id = CharField(max_length=12)

    class Meta:
        table_name = "user"
        schema = "system"

class UserGrade(BaseModel):
    id = CharField(primary_key=True, max_length=12)
    name = CharField(max_length=254)

    class Meta:
        table_name = "user_grade"
        schema = "system"

class WorkLog(BaseModel):
    id = CharField(primary_key=True, max_length=12)
    ticket_id = CharField(max_length=12)
    spent_time = FloatField()
    public_spent_time = FloatField()
    comment = TextField()

    class Meta:
        table_name = "work_log"
        schema = "sd"

def get_tickets(last_sync_date=None):
    if last_sync_date:
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
            .where((Ticket.ticket_state_id == 'SYS_10') & (Ticket.create_date > last_sync_date))
            .dicts()
            .iterator()
        )
    else:
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
            .where(Ticket.ticket_state_id == 'SYS_10')
            .dicts()
            .iterator()
        )

    return tickets