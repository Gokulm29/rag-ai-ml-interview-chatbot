from database import (
    connect_database,
    create_conversation,
    save_message,
    get_all_conversations,
    get_conversation_messages,
)

# 1. Connect
connect_database()

# 2. Create conversation
chat_id = create_conversation(
    "Linear Regression Discussion"
)

print("\nCreated Chat ID:", chat_id)

# 3. Save user message
save_message(
    chat_id,
    "user",
    "What is Linear Regression?"
)

# 4. Save assistant message
save_message(
    chat_id,
    "assistant",
    "Linear Regression predicts continuous values."
)

# 5. Get conversations
print("\nConversation History:")

for chat in get_all_conversations():
    print(chat["title"])

# 6. Load messages
print("\nMessages:")

for message in get_conversation_messages(chat_id):
    print(message["role"], ":", message["content"])