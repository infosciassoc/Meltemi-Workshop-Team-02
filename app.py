import datetime

import requests
import streamlit as st

# FastAPI server details
API_URL = "http://127.0.0.1:8000/chat"
START_CONVERSATION_URL = "http://127.0.0.1:8000/start_conversation"
ALL_CONVERSATIONS_URL = "http://127.0.0.1:8000/all_conversations"
HISTORY_URL = "http://127.0.0.1:8000/history"
STATUS_URL = "http://127.0.0.1:8000/status"
# Retry parameters
RETRY_TIMEOUT = 10  # Total timeout in seconds
RETRY_ATTEMPTS = 5  # Interval between retries in seconds


# Generate a localized welcome message
def generate_welcome_message():
	dt = datetime.datetime.now()
	return f"{'Καλημέρα' if dt.hour < 12 else 'Καλησπέρα'}. Τι καλό θα ήθελες να φτιάξουμε;"


# Function to start a new conversation
def start_new_conversation():
	try:
		response = requests.post(START_CONVERSATION_URL)
		response.raise_for_status()
		return response.json().get("conversation_id")
	except requests.exceptions.RequestException:
		st.error("Failed to start a new conversation. Please try again.")
		return None


# Function to fetch all conversations
def fetch_all_conversations():
	try:
		response = requests.get(ALL_CONVERSATIONS_URL)
		response.raise_for_status()
		return response.json().get("conversations", [])
	except requests.exceptions.RequestException:
		st.error("Failed to fetch conversations. Please try again.")
		return []


# Function to fetch a conversation history
def fetch_conversation_history(conversation_id):
	try:
		response = requests.get(f"{HISTORY_URL}/{conversation_id}")
		response.raise_for_status()
		return response.json().get("history", [])
	except requests.exceptions.RequestException:
		st.error("Failed to fetch conversation history. Please try again.")
		return []


# Function to send a user message to the server
def query_fastapi(user_message, conversation_id):
	try:
		payload = {
			"conversation_id": conversation_id,
			"user_message": user_message,
			"messages": st.session_state.messages
		}
		response = requests.post(API_URL, json=payload)
		response.raise_for_status()
		return response.json().get("response", "No response from server.")
	except requests.exceptions.RequestException:
		return "I'm having trouble connecting to the server. Please try again later."


# Initialize session state
if "conversation_id" not in st.session_state:
	# Start a new conversation when the app loads
	new_conversation_id = start_new_conversation()
	if new_conversation_id:
		st.session_state.conversation_id = new_conversation_id
		st.session_state.messages = [{"role": "assistant", "content": generate_welcome_message()}]

if "messages" not in st.session_state:
	st.session_state.messages = [{"role": "assistant", "content": generate_welcome_message()}]

# Sidebar: List all conversations
with st.sidebar:
	st.title("Conversations")
	conversations = fetch_all_conversations()

	# Display existing conversations in the sidebar
	if conversations:
		for convo in conversations:
			convo_time = datetime.datetime.fromtimestamp(convo["start_time"]).strftime('%Y-%m-%d %H:%M:%S')
			# Use unique keys to differentiate buttons
			if st.button(f"Conversation started at {convo_time}", key=f"select_{convo['id']}"):
				# Switch to the selected conversation
				st.session_state.conversation_id = convo["id"]
				st.session_state.messages = fetch_conversation_history(convo["id"])
	else:
		st.write("No previous conversations.")

	# Button to start a new conversation
	if st.button("Start a New Conversation", key="new_conversation"):
		new_conversation_id = start_new_conversation()
		if new_conversation_id:
			# Set up a new conversation with a fresh ID
			st.session_state.conversation_id = new_conversation_id
			st.session_state.messages = [{"role": "assistant", "content": generate_welcome_message()}]

# Main UI
st.title("mAICookbook")

# Display chat history
for message in st.session_state.messages:
	if message["role"] == "user":
		st.chat_message("user").write(message["content"])
	else:
		st.chat_message("assistant").write(message["content"])

# Input box for user message
if st.session_state.conversation_id:
	if user_input := st.chat_input("Ask a question or say something:"):
		# Add user message to chat history
		st.session_state.messages.append({"role": "user", "content": user_input})
		st.chat_message("user").write(user_input)

		# Query the FastAPI server
		assistant_response = query_fastapi(user_input, st.session_state.conversation_id)

		# Add assistant message to chat history
		st.session_state.messages.append({"role": "assistant", "content": assistant_response})
		st.chat_message("assistant").write(assistant_response)
else:
	st.write("Initializing a new conversation...")
