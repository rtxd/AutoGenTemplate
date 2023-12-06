# Standard library imports
import json
import os

# Third-party imports
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import openai
import requests
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities.google_serper import GoogleSerperAPIWrapper

# Local application/library specific imports
import autogen
from autogen import config_list_from_json
from autogen import UserProxyAgent
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent

# Load our own modules if required
#from search_utils import google_search

# --------------------------------------------------------------
# Load environment variables
# --------------------------------------------------------------
load_dotenv()
airtable_api_key = os.getenv("AIRTABLE_API_KEY")
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
config_list = config_list_from_json("OAI_CONFIG_LIST")
openai_api_key = os.getenv("OPENAI_API_KEY")

# --------------------------------------------------------------
# Main App
# --------------------------------------------------------------

openai.api_key = openai_api_key
OAIClient = openai.OpenAI()

# --------------------------------------------------------------
# google_search: uses langchain wrapper on serper.dev
# this function will be used by our searchAssistant agent
# --------------------------------------------------------------
def google_search(search_keyword):    
    search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
    results = search.results(search_keyword)
    print("google_search: ",search_keyword,results)
    return results

# --------------------------------------------------------------
# upload_file uploads a local document to OpenAI that can then be 
# provided to our docRetrievalAssistant (via file_id) to access 
# --------------------------------------------------------------
def upload_file(path):
    # Upload a file with an "assistants" purpose
    file = OAIClient.files.create(file=open(path, "rb"), purpose="assistants")
    return file

# --------------------------------------------------------------
# One off use of upload_file to get our docs up - uncomment when you need to upload files
# --------------------------------------------------------------
#script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
#rel_path = "data\Transcript 1 - 10 JavaScript changes you missed in 2023 by Fireship.txt"
#rel_path = "data\Transcript 2 - How programmers flex on each other by Fireship.txt"
#rel_path = "data\Transcript 3 - Masterclass- AI-driven Development for Programmers by Fireship.txt"
#abs_file_path = os.path.join(script_dir, rel_path)
#print(abs_file_path)
#file = upload_file(abs_file_path)
#print("file id:",file.id)
#quit()

# --------------------------------------------------------------
# Create Agents
# --------------------------------------------------------------

# ------------------------------------------------------- #
# Create user proxy agent 
user_proxy = UserProxyAgent(name="user_proxy",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=1
    )

# ------------------------------------------------------- #
# Create researcher agent - this is provided as an example only for accessing an assistant that already exists
#researcher = GPTAssistantAgent(
#    name = "researcher",
#    llm_config = {
#        "config_list": config_list,
#        "assistant_id": "asst_p5VK6qWA5XD1GaIuohxP39UX"
#    }
#)

# ------------------------------------------------------- #
# Create retrieval agent 
retrievalAssistant = GPTAssistantAgent(
        name="retrievalAssistant",
        instructions="""You're an expert tweet writer.
          Use your knowledge base to best write tweets based on the cutomer topics.
          Be friendly and funny.""",
        llm_config={
            "config_list": config_list,
            "assistant_id": None,
            "tools": [{"type": "retrieval"}],
            "file_ids": ["file-WpfPsN8NA3Dr3eS1ObKGWPwM"]
        }
)

# ------------------------------------------------------- #
# Create a search agent - In 3 parts
# 1) we're going to define the google_search function the search agent can use
# The desciptions let the agent know how to use the function

tools_list = [{
    "type": "function",
    "function": {
        "name": "google_search",
        "description": "Use google search to return the Google search results obtained for a search keywords",
        "parameters": {
            "type": "object",
            "properties": {
                "search_keyword": {
                    "type": "string",
                    "description": "Good keywords that are likely to return the results for what you are looking for"
                }
            },
            "required": ["search_keyword"]
        }
    }
}]

# 2) Create the search agent with knowledge of how to use the function provided above
searchAssistant = GPTAssistantAgent(
    name="searchAssistant",
    instructions="""
    You are a world-class research assistant who can do detailed research on any topic and produce facts based results
    you do not make things up, you will try as hard as possible to gather facts & data to back up the research
    Please make sure you compelte the objective above with the following rules:
    You should do enough research to gather as much information as possible about the objective.
    Reply TERMINATE when you have finished your search.
    """,
    #If there are url of relevant links & articles, you will scrape it to gather more information\n3. After scrapping & search, you should think 'is there any new things I should search & scrape baed on the data i collected to increase research quality?' If the answer is yes, continue. But don't do it more than 3 iterations\n4. You should not make things up. You should only write facts & data you have gathered\n5. In the final output, you should include all reference data & links to back up your research. \n6. Do not use LinkedIn or any other social media, they are mostly out dated data

    llm_config={
        "config_list": config_list,
        "assistant_id": None,
        "tools": tools_list,
    })

# 3) Map the function we've described to the agent to the real function in our code.
searchAssistant.register_function(
    function_map={
        "google_search": google_search,
    }
)

# --------------------------------------------------------------
# Use autogen to have the agents chat to each other 
# --------------------------------------------------------------

#---------------------------------------------------------------
# Scenario 1 - we're going to ask the retrievalAgent to create tweets based on Tweet_Like_a_Pro.pdf

# Create group chat
groupchat = autogen.GroupChat(agents=[user_proxy, retrievalAssistant], messages=[], max_round=10)
group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

#message = """Using the suggesting recommended in the document Tweet_Like_a_Pro
#create a tweet for the following news:
#The UK has published the world’s first global guidelines for securing AI systems against cyberattacks. The new guidelines aim to ensure AI technology is developed safely and securely.
#The guidelines were developed by the UK’s National Cyber Security Centre (NCSC) and the US’ Cybersecurity and Infrastructure Security Agency (CISA). They have already secured endorsements from 17 other countries, including all G7 members.
#The guidelines provide recommendations for developers and organisations using AI to incorporate cybersecurity at every stage. This “secure by design” approach advises baking in security from the initial design phase through development, deployment, and ongoing operations.  
#Specific guidelines cover four key areas: secure design, secure development, secure deployment, and secure operation and maintenance. They suggest security behaviours and best practices for each phase.
#The launch event in London convened over 100 industry, government, and international partners. Speakers included reps from Microsoft, the Alan Turing Institute, and cyber agencies from the US, Canada, Germany, and the UK.  
#"""

message = """Provide a concise summary of the article Tweet Like a Pro
"""

user_proxy.initiate_chat(group_chat_manager, message=message)
quit()


#---------------------------------------------------------------
# Scenario 2 - we're going to ask the searchAgent to search the answer to a question on-line

# Create group chat
#groupchat = autogen.GroupChat(agents=[user_proxy, searchAssistant], messages=[], max_round=10)
#group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

#message = "Can you find out who won the F1 in 2023 and give me a few details about it."

#user_proxy.initiate_chat(group_chat_manager, message=message)
#quit()



# Function for scraping - will need this for later
def summary(objective, content):
    llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo-16k-0613")
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size = 10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "objective"])
    summary_chain = load_summarize_chain(
        llm=llm, 
        chain_type='map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt = map_prompt_template,
        verbose = False
    )
    return summary_chain.run(input_documents=docs, objective=objective)

