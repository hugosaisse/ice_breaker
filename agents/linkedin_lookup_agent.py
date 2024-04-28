import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)
from langchain import hub
from tools.tools import get_profile_url_tavily

def lookup(name: str) -> str:
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
    )
    template = """
    Given the full name {name_of_person}, can you find the LinkedIn profile of this person?
    Your answer should be only a URL to the LinkedIn profile.
    """
    prompt_template = PromptTemplate(
        template=template,
        input_variables=["name_of_person"],
    )
    tools_for_agent = [
        Tool(
            name="crawl google for linkedin profile page",
            function=get_profile_url_tavily,
            description="useful when you need to find a linkedin profile page URL",
        )
    ]
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(
        llm=llm,
        tools=tools_for_agent,
        prompt=react_prompt,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}    
    )

    linkedin_profile_url = result["output"]
    
    if __name__ == "__main__":
        linkedin_url = lookup(name="Hugo Mentzingen")
        print(linkedin_url)