import autogen
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json, OpenAIWrapper


load_dotenv()
config_list = config_list_from_json("OAI_CONFIG_LIST")


assistant = AssistantAgent(name="Assistant", llm_config=config_list)

