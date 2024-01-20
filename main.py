from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

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

if __name__ == "__main__":
    print(generate_girl_name("Karnataka", "A beautiful pretty girl who is innocent and cute and her name starts with A and ends with A"))
    