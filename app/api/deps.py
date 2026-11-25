from app.db.session import get_session
from app.llm.client import get_llm_client


def get_db():
    yield from get_session()


def get_llm():
    return get_llm_client()
