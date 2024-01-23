from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import faiss

#discord libraries

import discord
from dotenv import load_dotenv
from discord.ext import commands
import os
import pickle

 
load_dotenv()
 
embeddings = OpenAIEmbeddings()
 
 

def create_vector_db_from_youtube(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                    chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    
    db = FAISS.from_documents(docs, embeddings)
    
    return db

def load_or_create_vector_db(video_url: str, db_file_path: str) -> FAISS:
    # Check if the file with the result exists
    if os.path.exists(db_file_path):
        # Load the existing FAISS index from the file
        db = faiss.read_index(db_file_path)
    else:
        # Perform the operation and create the FAISS index
        db = create_vector_db_from_youtube(video_url)

        # Save the FAISS index to the file
        faiss.write_index(db, db_file_path)

    return db
    if os.path.exists(db_file_path):
        # Load the existing result from the file
        with open(db_file_path, 'rb') as file:
            db = pickle.load(file)
    else:
        # Perform the operation and create the vector db
        db = create_vector_db_from_youtube(video_url)

        # Save the result to the file
        with open(db_file_path, 'wb') as file:
            pickle.dump(db, file)

    return db
 
def get_response_from_query(db, query, k=4):
    #text-davinci can handle 4097 tokens
    
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    
    llm = OpenAI(temperature=0.8)
    
    prompt = PromptTemplate(
        input_variables=["question","docs"],
        template=""" You are the helpful smart friend of me that can answer questions about the recent videos we have watched based 
                     on the video's transcript.set
                     
                     Answer the following question: {question}
                     By searching the following video transcript: {docs}
                     
                     Mainly use the factual information from the transcript but you can add your opinion too.
                     
                     If you feel like you don't have enough information to answer the question, try to come up with the 
                     related suitable answer from the information you have.
                     
                     your answer should be detailed and long.
        """,
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response




# Define your intents
intents = discord.Intents.default()
intents.message_content = True #v2
client = commands.Bot(command_prefix = '!' , intents=intents)

# Access the bot token from the environment variables
bot_token = os.getenv('DISCORD_TOKEN')

@client.event
async def on_ready():
    print("The bot is ready for use")
    print("-------------------------")

@client.event
async def on_message(message):
    # Check if the message starts with "!"
    if message.content.startswith('!'):
        # Remove the first "!" symbol and get the remaining message
        remaining_message = message.content[1:]
        
        db_file_path = "vector_db.pkl"
        video_url="https://www.youtube.com/watch?v=3JqgJLiX0MQ" 
        db=load_or_create_vector_db(video_url,db_file_path)
        response = get_response_from_query(db,remaining_message)
                
        # Reply to the message
        await message.channel.send(response)

    # Process other commands
    await client.process_commands(message)

@client.command()
async def ilove(ctx):
    await ctx.send("I Love You Too and I Love You More!")
    
client.run(bot_token)

