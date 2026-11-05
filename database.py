import os
import hashlib
import hmac
import secrets
from datetime import datetime, timezone

from bson import ObjectId
from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError, DuplicateKeyError
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = "rag_interview_db"

client = None
db = None

# PBKDF2 is available in Python's standard library; no extra password package required.
_PASSWORD_ITERATIONS = 600_000


def _now():
    return datetime.now(timezone.utc)


def _normalise_email(email):
    return str(email or "").strip().lower()


def _hash_password(password, salt=None):
    if not isinstance(password, str) or len(password) < 10:
        raise ValueError("Password must contain at least 10 characters")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PASSWORD_ITERATIONS
    )
    return {
        "password_salt": salt.hex(),
        "password_hash": digest.hex(),
        "password_iterations": _PASSWORD_ITERATIONS,
    }


def _verify_password(password, user):
    try:
        salt = bytes.fromhex(user["password_salt"])
        iterations = int(user.get("password_iterations", _PASSWORD_ITERATIONS))
        expected = bytes.fromhex(user["password_hash"])
        actual = hashlib.pbkdf2_hmac(
            "sha256", str(password).encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(actual, expected)
    except (KeyError, TypeError, ValueError):
        return False


# ==========================================
# DATABASE CONNECTION
# ==========================================

def connect_database():
    global client, db
    if not MONGODB_URI:
        raise ValueError("MONGODB_URI is missing from environment / Streamlit Secrets")
    try:
        new_client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=15000,
            connectTimeoutMS=15000,
            appname="InterviewAI",
        )
        new_client.admin.command("ping")
        database = new_client[DATABASE_NAME]

        database["users"].create_index("email", unique=True)
        database["users"].create_index("google_sub", unique=True, sparse=True)
        database["conversations"].create_index(
            [("user_id", 1), ("updated_at", DESCENDING)]
        )
        database["messages"].create_index(
            [("chat_id", 1), ("created_at", 1)]
        )

        client, db = new_client, database
        print("MongoDB Atlas connected successfully!")
        return db
    except PyMongoError as error:
        if "new_client" in locals():
            new_client.close()
        client = None
        db = None
        raise ConnectionError(f"MongoDB connection failed: {error}") from error


def get_database():
    global db
    if db is None:
        return connect_database()
    return db


def get_collections():
    database = get_database()
    return database["conversations"], database["messages"]


# ==========================================
# USER AUTHENTICATION
# ==========================================

def register_user(name, email, password):
    """Create an email/password account and return its MongoDB user ID."""
    name = str(name or "").strip()
    email = _normalise_email(email)
    if not name:
        raise ValueError("Name is required")
    if not email or "@" not in email:
        raise ValueError("Enter a valid email address")
    password_fields = _hash_password(password)
    users = get_database()["users"]
    document = {
        "name": name[:80],
        "email": email,
        "auth_provider": "password",
        "created_at": _now(),
        **password_fields,
    }
    try:
        result = users.insert_one(document)
        return str(result.inserted_id)
    except DuplicateKeyError as error:
        raise ValueError("An account with this email already exists") from error


def authenticate_user(email, password):
    """Return a safe user profile on success; otherwise return None."""
    email = _normalise_email(email)
    user = get_database()["users"].find_one({"email": email})
    if not user or not _verify_password(password, user):
        return None
    return {
        "user_id": str(user["_id"]),
        "name": user.get("name", "User"),
        "email": user["email"],
        "auth_provider": user.get("auth_provider", "password"),
    }


def get_or_create_google_user(google_sub, email, name):
    """Link/create an OIDC Google identity using the stable OIDC subject."""
    google_sub = str(google_sub or "").strip()
    email = _normalise_email(email)
    name = str(name or "Google user").strip()[:80]
    if not google_sub or not email or "@" not in email:
        raise ValueError("Google did not provide a valid verified identity")

    users = get_database()["users"]
    user = users.find_one({"google_sub": google_sub})
    if user:
        return {
            "user_id": str(user["_id"]), "name": user.get("name", name),
            "email": user["email"], "auth_provider": user.get("auth_provider", "google")
        }

    # Do not silently link a Google identity to an existing password account by email.
    if users.find_one({"email": email}):
        raise ValueError(
            "An account already exists for this email. Sign in with that account first; "
            "automatic account linking is disabled for security."
        )

    result = users.insert_one({
        "name": name,
        "email": email,
        "google_sub": google_sub,
        "auth_provider": "google",
        "created_at": _now(),
    })
    return {"user_id": str(result.inserted_id), "name": name,
            "email": email, "auth_provider": "google"}


def get_user_profile(user_id):
    if not user_id or not ObjectId.is_valid(str(user_id)):
        return None
    user = get_database()["users"].find_one({"_id": ObjectId(str(user_id))})
    if not user:
        return None
    return {"user_id": str(user["_id"]), "name": user.get("name", "User"),
            "email": user.get("email", ""),
            "auth_provider": user.get("auth_provider", "password")}


# ==========================================
# CONVERSATIONS - OWNER SCOPED
# ==========================================

def create_conversation(title="New Chat", user_id=None):
    if not user_id or not ObjectId.is_valid(str(user_id)):
        raise ValueError("A valid authenticated user_id is required")
    conversations, _ = get_collections()
    now = _now()
    result = conversations.insert_one({
        "user_id": str(user_id),
        "title": str(title or "New Chat")[:120],
        "pinned": False,
        "created_at": now,
        "updated_at": now,
    })
    return str(result.inserted_id)


def get_all_conversations(user_id=None):
    if not user_id:
        raise ValueError("A valid authenticated user_id is required")
    conversations, _ = get_collections()
    return list(conversations.find({"user_id": str(user_id)}).sort(
        [("pinned", DESCENDING), ("updated_at", DESCENDING)]
    ))


def _owned_conversation(chat_id, user_id):
    if not user_id or not ObjectId.is_valid(str(chat_id)):
        return None
    conversations, _ = get_collections()
    return conversations.find_one({"_id": ObjectId(str(chat_id)), "user_id": str(user_id)})


def get_conversation_messages(chat_id, user_id=None):
    if not _owned_conversation(chat_id, user_id):
        raise PermissionError("Conversation not found or access denied")
    _, messages = get_collections()
    return list(messages.find({"chat_id": str(chat_id)}, {"_id": 0}).sort("created_at", 1))


def save_message(chat_id, role, content, sources=None, user_id=None):
    if role not in {"user", "assistant"}:
        raise ValueError("Invalid message role")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Message content cannot be empty")
    conversation = _owned_conversation(chat_id, user_id)
    if not conversation:
        raise PermissionError("Conversation not found or access denied")

    conversations, messages = get_collections()
    now = _now()
    messages.insert_one({
        "chat_id": str(chat_id),
        "user_id": str(user_id),
        "role": role,
        "content": content,
        "sources": sources or [],
        "created_at": now,
    })
    update_fields = {"updated_at": now}
    if role == "user" and conversation.get("title") == "New Chat":
        update_fields["title"] = content[:50]
    conversations.update_one(
        {"_id": ObjectId(str(chat_id)), "user_id": str(user_id)},
        {"$set": update_fields},
    )


def update_chat_title(chat_id, title, user_id=None):
    if not title or not str(title).strip():
        raise ValueError("Chat title cannot be empty")
    conversations, _ = get_collections()
    result = conversations.update_one(
        {"_id": ObjectId(str(chat_id)), "user_id": str(user_id)},
        {"$set": {"title": str(title).strip()[:120], "updated_at": _now()}},
    ) if user_id and ObjectId.is_valid(str(chat_id)) else None
    if not result or result.matched_count != 1:
        raise PermissionError("Conversation not found or access denied")


def update_pin_status(chat_id, pinned, user_id=None):
    conversations, _ = get_collections()
    result = conversations.update_one(
        {"_id": ObjectId(str(chat_id)), "user_id": str(user_id)},
        {"$set": {"pinned": bool(pinned), "updated_at": _now()}},
    ) if user_id and ObjectId.is_valid(str(chat_id)) else None
    if not result or result.matched_count != 1:
        raise PermissionError("Conversation not found or access denied")


def delete_conversation(chat_id, user_id=None):
    if not user_id or not ObjectId.is_valid(str(chat_id)):
        raise PermissionError("Conversation not found or access denied")
    conversations, messages = get_collections()
    result = conversations.delete_one({
        "_id": ObjectId(str(chat_id)), "user_id": str(user_id)
    })
    if result.deleted_count != 1:
        raise PermissionError("Conversation not found or access denied")
    messages.delete_many({"chat_id": str(chat_id), "user_id": str(user_id)})


def close_database():
    global client, db
    if client:
        client.close()
    client = None
    db = None
