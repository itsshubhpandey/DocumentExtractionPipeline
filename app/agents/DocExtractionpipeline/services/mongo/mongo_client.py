import os

from dotenv import load_dotenv

from app.infrastructure.motor_database import get_motor_client
 
load_dotenv()


def _get_database():
    mongo_uri = os.getenv("MONGODB_URL")
    mongo_database = os.getenv("MONGODB_DATABASE")
    mongo_timeout_ms = int(os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "5000"))

    missing_settings = [
        name
        for name, value in {
            "MONGODB_URL": mongo_uri,
            "MONGODB_DATABASE": mongo_database,
        }.items()
        if not value
    ]

    if missing_settings:
        raise ValueError(
            "Missing required MongoDB environment variable(s): "
            + ", ".join(missing_settings)
        )

    client = get_motor_client()
    return client[mongo_database]


async def get_document(collection_name, filter_query):
    db = _get_database()
    collection = db[collection_name]
    return await collection.find_one(filter_query)
