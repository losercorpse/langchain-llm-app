from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
 
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


video_url="https://www.youtube.com/watch?v=3JqgJLiX0MQ" 
db=create_vector_db_from_youtube(video_url)
response = get_response_from_query(db,"why is aashalad is so smart?")
print(response)
