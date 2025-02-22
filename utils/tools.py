from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from langchain.chains import LLMChain
from langchain.agents import create_sql_agent, initialize_agent
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool
from langchain.tools import tool
from utils.llm_setup import llm

@tool
def greeting() -> str:
    """Caller initiates the conversation with a greeting (e.g., "Hello," "How may I help you?"). The function responds with a standard greeting (e.g., "Hello, thank you for calling, how may I assist you today?")."""
    return "Hello, thank you for calling, how may I assist you today?"

@tool
def customerdatabase(user_prompt: str) -> str :
    """ Using the prompt entered by user to check the database for their data and updating their data.
    - Search answer for questions or concerns regarding packages, date and time,status or payment methods. 
     -Agent accesses the customers's account information, clarifies billing details, explains packages, and assists with payment-related issues or disputes."""

    conn_str = "sqlite:///files/CustomerData.db"
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

    query = agent_executor.invoke(f'{prompt_template} given the table name is CustomerData?')

    return query


@tool
def company_policies(prompt: str) -> str:
    """Search answer for questions or concerns regarding order cancellation policy, cancellation of orders
    Search answer for questions or concerns regarding refund policy, refunding questions
    Search details for any packages/products/services provided"""

    # load documents
    documents = SimpleDirectoryReader("files/companypolicies").load_data()
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
    func=customerdatabase,
    description="This contains all the information about customer data like packages, order date and time and status. Any updates to customer database can be made here"
)

policy_inquiry = Tool(
    name="company_policies_products",
    func=company_policies,
    description="This contains all the company policies like refund and exchange policies and packages, products, and services details",
)

customer_greeting = Tool(
    name="greeting_customers",
    func=greeting,
    description="This tells us how to greet a customer when they say hi or initialte the conversation",
)

tools=[customer_greeting, policy_inquiry,customer_inquiry]

PREFIX = "<<SYS>> Select a function from the given tools based on the user's query. Only one tool per query. <</SYS>>\n"

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    max_iterations=4,
    agent_kwargs={'prefix': PREFIX}
)