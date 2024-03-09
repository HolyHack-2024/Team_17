from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage,SystemMessage
from langchain.agents import AgentExecutor
import os
import pandas as pd
import numpy as np



#df = pd.read_csv("cosmetics.csv")

user_input = input("Ask me anything concerning your skin care \n")

user_skin_conditions = {
    "skin_tone": "medium",
    "acne_concern_level": "high",
    "skin_type": "oily"
}

chat_model = AzureChatOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
    api_version="2023-05-15", 
    model="gpt-4-32k"
   ) 



messages = [
    SystemMessage(
        content="You are a skin scare expert that can give advice on any skin care related goal or problem. You will receive a prompt from the user, presenting his request and a set of string representing his current condition skin condition : skin tone, acne concern level, and skin type."
    ),
    HumanMessage(
        content="The user has the following request: "+ user_input + "His current situation is :"+ "Skin Tone: " + user_skin_conditions['skin_tone'] + ", Acne Concern Level: " + user_skin_conditions['acne_concern_level'] + ", Skin Type: " + user_skin_conditions['skin_type'] 
    ),
]  

print(chat_model.invoke(messages).content)
