import os
from datetime import datetime, timezone

from bson import ObjectId
from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError
from dotenv import load_dotenv


load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = "rag_interview_db"

client = None
db = None


# ==========================================
# DATABASE CONNECTION
# ==========================================

def connect_database():

    global client, db

    if not MONGODB_URI:
        raise ValueError("MONGODB_URI is missing in .env")

    try:
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=10000
        )

        client.admin.command("ping")

        db = client[DATABASE_NAME]

        # Collections
        db["conversations"].create_index(
            [("updated_at", DESCENDING)]
        )

        db["messages"].create_index(
            [("chat_id", 1), ("created_at", 1)]
        )

        print("MongoDB Atlas connected successfully!")

        return db

    except PyMongoError as error:
        client = None
        db = None

        raise ConnectionError(
            f"MongoDB connection failed: {error}"
        ) from error


def get_database():

    global db

    if db is None:
        return connect_database()

    return db


# ==========================================
# COLLECTIONS
# ==========================================

def get_collections():

    database = get_database()

    conversations = database["conversations"]
    messages = database["messages"]

    return conversations, messages


# ==========================================
# CREATE NEW CONVERSATION
# ==========================================

def create_conversation(title="New Chat"):

    conversations, _ = get_collections()

    now = datetime.now(timezone.utc)

    result = conversations.insert_one({
        "title": title,
        "created_at": now,
        "updated_at": now
    })

    return str(result.inserted_id)


# ==========================================
# SAVE MESSAGE
# ==========================================

def save_message(chat_id, role, content, sources=None):

    if role not in {"user", "assistant"}:
        raise ValueError("Invalid message role")

    if not content or not content.strip():
        raise ValueError("Message content cannot be empty")

    conversations, messages = get_collections()

    now = datetime.now(timezone.utc)

    messages.insert_one({
        "chat_id": chat_id,
        "role": role,
        "content": content,
        "sources": sources or [],
        "created_at": now
    })

    update_fields = {
        "updated_at": now
    }

    # Use the first user message as the chat title.
    if role == "user":
        conversation = conversations.find_one(
            {"_id": ObjectId(chat_id)}
        )

        if conversation and conversation.get("title") == "New Chat":
            update_fields["title"] = content[:50]

    conversations.update_one(
        {"_id": ObjectId(chat_id)},
        {"$set": update_fields}
    )


# ==========================================
# GET ALL CONVERSATIONS
# ==========================================

def get_all_conversations():

    conversations, _ = get_collections()

    return list(
        conversations.find().sort(
            "updated_at",
            DESCENDING
        )
    )


# ==========================================
# LOAD CONVERSATION MESSAGES
# ==========================================

def get_conversation_messages(chat_id):

    _, messages = get_collections()

    return list(
        messages.find(
            {"chat_id": chat_id},
            {"_id": 0}
        ).sort("created_at", 1)
    )


# ==========================================
# DELETE CONVERSATION
# ==========================================

def delete_conversation(chat_id):

    conversations, messages = get_collections()

    conversations.delete_one({
        "_id": ObjectId(chat_id)
    })

    messages.delete_many({
        "chat_id": chat_id
    })


# ==========================================
# CLOSE CONNECTION
# ==========================================

def close_database():

    global client, db

    if client:
        client.close()

    client = None
    db = None