import openai
import autogen
import json
import os
import requests

from autogen import config_list_from_json
from autogen import UserProxyAgent
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
airtable_api_key = os.getenv("AIRTABLE_API_KEY")
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
config_list = config_list_from_json("OAI_CONFIG_LIST")
