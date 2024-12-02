import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Union, Generator

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.litellm import LiteLLM
from pydantic import BaseModel

from prompts import text_qa_prompt, text_refine_prompt


class Message(BaseModel):
	role: str
	content: str


class RequestBody(BaseModel):
	conversation_id: str
	user_message: str
	messages: List[Message]


class Server:
	def __init__(self):
		self.documents = None
		self.index = None
		self.embed_model = None
		self.llm = None
		self.conversations = []
		load_dotenv()

	async def on_startup(self):
		self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
		documents = self.load_documents()
		nodes = self.chunk_documents(documents, batch_size=32)

		print("Building index...")
		self.index = VectorStoreIndex(nodes, embed_model=self.embed_model)

		self.llm = LiteLLM(
			"hosted_vllm/meltemi-vllm",
			api_base=os.getenv("API_BASE"),
			api_key=os.getenv("API_KEY")
		)

	async def on_shutdown(self):
		pass

	def start_new_conversation(self) -> str:
		"""Start a new conversation and return its unique ID."""
		conversation_id = str(uuid.uuid4())
		self.conversations.append({"id": conversation_id, "start_time": time.time(), "messages": []})
		# Sort conversations by start time (latest first)
		self.conversations.sort(key=lambda c: c["start_time"], reverse=True)
		return conversation_id

	def chunk_documents(self, documents, batch_size=32):
		def batch(iterable, n=1):
			it = iter(iterable)
			from itertools import islice
			while chunk := list(islice(it, n)):
				yield chunk


		splitter = SemanticSplitterNodeParser(
		    buffer_size=3, breakpoint_percentile_threshold=90, embed_model=self.embed_model
		)

		nodes = []
		for doc_batch in batch(documents, batch_size):
			batch_nodes = splitter.get_nodes_from_documents(doc_batch)
			nodes.extend(batch_nodes)
		return nodes

	@staticmethod
	def load_documents():
		def recipe_text_template(row: pd.Series) -> str:
			template = (f"Η συνταγή για {row['name']} είναι ένα {row['Category']} που "
						f"χρειάζεται τα εξής υλικά: {row['Ingredients']}. ")
			if not pd.isna(row['Preparation Time']):
				template += f"Έχει χρόνο προετοιμασίας {row['Preparation Time']} "
			if not pd.isna(row['Total Time']):
				template += f"και συνολικά παίρνει {row['Total Time']}. "
			if not pd.isna(row['Number of Servings']):
				template += f"Οι μερίδες που φτιάχνει είναι {row['Number of Servings']}. "
			if not pd.isna(row['Keywords']):
				template += f"Χαρακτηριστικές λέξεις που περιγράφουν αυτή τη συνταγή είναι: {row['Keywords']}."
			if not pd.isna(row['Instructions']):
				template += f"Ο τρόπος προετοιμασίας είναι ο εξής: {row['Instructions']}."
			return template

		recipes = pd.read_csv("hf://datasets/Depie/Recipes_Greek/recipes_greek.csv").apply(recipe_text_template, axis=1)
		return [Document(text=t) for t in recipes.to_list()]

	def serve(self, user_message: str, conversation_id: str) -> Union[str, Generator]:
		# Add user message to conversation history
		self.store_message(conversation_id, Message(role="user", content=user_message))

		query_engine = self.index.as_query_engine(
			llm=self.llm, text_qa_prompt=text_qa_prompt, text_refine_prompt=text_refine_prompt, streaming=False
		)

		response = query_engine.query(user_message)

		# Add assistant response to conversation history
		assistant_message = Message(**{'role': "assistant", 'content': response.response})
		self.store_message(conversation_id, assistant_message)

		return response.response

	def store_message(self, conversation_id: str, message: Message):
		"""Store a message in the conversation history."""
		for conversation in self.conversations:
			if conversation["id"] == conversation_id:
				conversation["messages"].append(message.model_dump())
				return
		raise ValueError(f"Conversation ID {conversation_id} not found.")

	def get_conversation(self, conversation_id: str) -> List[Message]:
		# TODO make self.conversations a dict for faster lookup
		for conversation in self.conversations:
			if conversation["id"] == conversation_id:
				return conversation["messages"]
		raise ValueError(f"Conversation ID {conversation_id} not found.")

	def get_all_conversations(self) -> List[Dict]:
		return [{"id": c["id"], "start_time": c["start_time"]} for c in self.conversations]


server = Server()


@asynccontextmanager
async def lifespan(app: FastAPI):
	await server.on_startup()
	yield
	await server.on_shutdown()


app = FastAPI(docs_url="/docs", redoc_url=None, lifespan=lifespan)


@app.post("/chat")
async def query_pipeline(body: RequestBody):
	"""Endpoint to handle user queries."""
	try:
		conversation_id = body.conversation_id
		user_message = body.user_message
		response_gen = server.serve(user_message, conversation_id)
		return {"response": "".join(response_gen)}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@app.post("/start_conversation")
async def start_conversation():
	"""Endpoint to start a new conversation."""
	try:
		conversation_id = server.start_new_conversation()
		return {"conversation_id": conversation_id}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{conversation_id}")
async def get_conversation_history(conversation_id: str):
	"""Endpoint to retrieve the conversation history for a given conversation ID."""
	try:
		history = server.get_conversation(conversation_id)
		return {"history": history}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/all_conversations")
async def get_all_conversations():
	"""Endpoint to retrieve all conversations sorted by start time."""
	try:
		conversations = server.get_all_conversations()
		return {"conversations": conversations}
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
