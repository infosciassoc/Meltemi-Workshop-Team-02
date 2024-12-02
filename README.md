# mAICookbook

mAICookbook is a personal cookbook assistant designed to help you explore Greek recipes interactively. 
The project uses a simple Server-Client architecture.

The server is a FastAPI service that hosts a simple RAG system built with LlamaIndex and Meltemi,
a powerful Greek LLM. The knowledge base of the RAG system is powered by a 
[Greek Recipes Dataset](https://huggingface.co/datasets/Depie/Recipes_Greek), 
providing a rich collection of traditional and modern Greek recipes.

The client is a simple Streamlit chatbot UI that allows users to interact with the RAG system and explore recipes. 
## Setup and Installation

1. Clone the repository
```bash
git clone https://github.com/infosciassoc/Meltemi-Workshop-Team-02.git
cd mAICookbook
```
2. Create a virtual environment using conda
```bash
conda create -n maicookbook python=3.11
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Set up Environment Variables
Create a .env file in the project root and add the following:
```bash
API_BASE=<your-meltemi-api-url>
API_KEY=<your-meltemi-api-key>
```

5. Run the FastAPI server
```bash
uvicorn server:app --reload
```
6. Run the Streamlit UI
```bash
streamlit run app.py
```

## Usage
1. Open the Streamlit app in your browser at http://localhost:8501.
2. Start a conversation with the chatbot
3. You can see previous conversations in the sidebar on the left side.

## Contributors

-  [elzanou](https://github.com/elzanou)
- [aspil](https://github.com/aspil)
