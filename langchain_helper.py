from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# for agents
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType



load_dotenv()

def generate_girl_name(girl_city, girl_personality):
    llm = OpenAI(temperature=0.8)

    prompt_template_name = PromptTemplate(
        input_variables=['girl_city','girl_personality'],
        template="An Indian girl is from {girl_city} in india and her personaltiy is {girl_personality}. Suggest me five indian names that suits her."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name)

    response=name_chain({'girl_city': girl_city, 'girl_personality': girl_personality})

    return response


def langchain_agent():
    llm=OpenAI(temperature=0.9)
    
    tools = load_tools(["wikipedia","llm-math"], llm = llm)
    
    agent = initialize_agent(
        tools, llm , agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    
    result = agent.run(
        "What does the noraml indina girl look in a partner? What are the qualities she desire in a man?"
    )
    
    print(result)

if __name__ == "__main__":
    langchain_agent()
    #print(generate_girl_name("Karnataka", "A beautiful pretty girl who is innocent and cute and her name starts with A and ends with A"))
    