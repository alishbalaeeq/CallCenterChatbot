from flask import Flask, render_template, request
import os
from loguru import logger
import re
from transformers import BitsAndBytesConfig
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.agents import create_sql_agent
from pyngrok import ngrok
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from accelerate import Accelerator
from langchain.agents import Tool
from langchain.agents.initialize import initialize_agent
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/model/mistral-7b-instruct-v0.2.Q8_0.gguf",
    temperature=0.1,
    max_new_tokens = 256,
    context_window = 3900,
    n_ctx=2048,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt_template = PromptTemplate.from_template(template)

accelerator = Accelerator()
llm = accelerator.prepare(llm)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.chunk_size=1024
Settings.llm = llm
Settings.embed_model = embed_model

@tool
def greeting() -> str:
    """Caller initiates the conversation with a greeting (e.g., "Hello," "How may I help you?"). The function responds with a standard greeting (e.g., "Hello, thank you for calling, how may I assist you today?")."""
    return "Hello, thank you for calling, how may I assist you today?"

@tool
def CustomerDatabase(user_prompt: str) -> str :
    """ Using the prompt entered by user to check the database for their data and updating their data.
    - Search answer for questions or concerns regarding packages, date and time,status or payment methods. 
     -Agent accesses the customers's account information, clarifies billing details, explains packages, and assists with payment-related issues or disputes."""

    conn_str = "sqlite:///database/CustomerData.db"
    db = SQLDatabase.from_uri(conn_str)

    PREFIX = "<<SYS>> You have a Table named 'CustomerData' with the following columns: " \
    "CustomerID, CustomerName, CustomerEmail, CustomerAge, CustomerGender, Product, " \
    "Payment_Method, Order_Time, Order_Date, and Status." \
    "- Only generate sql query for the given prompt <</SYS>>\n"

    agent_executor = create_sql_agent(
      llm=llm,
      toolkit=SQLDatabaseToolkit(db=db, llm=llm),
      verbose=True,
      agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
      agent_kwargs={
        'prefix': PREFIX,
        }
      )

    query = agent_executor.invoke(
                                      f'{prompt_template} given the table name is CustomerData?'
                                    )

    return query


@tool
def Company_Policies(prompt: str) -> str:
    """Search answer for questions or concerns regarding order cancellation policy, cancellation of orders
    Search answer for questions or concerns regarding refund policy, refunding questions
    Search details for any packages/products/services provided"""

    # load documents
    documents = SimpleDirectoryReader("data/companypolicies").load_data()
    Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)],
    )

    query_engine = index.as_query_engine()
    
    response = query_engine.query(prompt)
    
    return response


customer_inquiry = Tool(
    name="customer_inquiry",
    func=CustomerDatabase,
    description="This contains all the information about customer data like packages, order date and time and status. Any updates to customer database can be made here"
)

policy_inquiry = Tool(
    name="company_policies_products",
    func=Company_Policies,
    description="This contains all the company policies like refund and exchange policies and packages, products, and services details",
)

customer_greeting = Tool(
    name="greeting_customers",
    func=greeting,
    description="This tells us how to greet a customer when they say hi or initialte the conversation",
)

tools=[customer_greeting, policy_inquiry,customer_inquiry]

PREFIX = "<<SYS>> You are smart agent that selects a function from list of given functions based on user queries.\
Run only one function tool at a time or in one query.<</SYS>>\n"

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    max_iterations = 4,
    agent_kwargs={
        'prefix': PREFIX,
    }
)

logfile = "output.log"

logger.add(logfile, colorize=True, enqueue=True)
handler_1 = FileCallbackHandler(logfile)
handler_2 = StdOutCallbackHandler()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        logger.info(user_input)

        try:
            response = agent.invoke(user_input,{"callbacks": [handler_1, handler_2]})
        except Exception as e:
            response = 'An unexpected error occured, please try later'
        logger.info(response)

        return render_template('index.html', user_prompt=user_input, generated_code=response)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)