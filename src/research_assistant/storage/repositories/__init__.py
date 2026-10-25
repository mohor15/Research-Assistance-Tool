"""Repository/data-access layer — persistence/query infrastructure only.

No LLM logic, no scraping logic, no business reasoning lives here: each
repository just maps a Pydantic domain model to/from its SQLite table.
"""
