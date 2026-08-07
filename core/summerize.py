from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os

#This function initializes and returns a ChatMistralAI instance with the specified model and API key.
def getllm():
    return ChatMistralAI(
        model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"), temperature = 0.3)

#This function creates a chat prompt template for summarization tasks.
def split_transcript(transcript: str) -> list:
    "Split the transcript into smaller segments for summarization."
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200,
    )
    return text_splitter.split_text(transcript)


#This function summarizes the transcript using the Mistral AI model.
def summarize_transcript(transcript: str) -> str:
    "Summarize the transcript using the Mistral AI model."
    llm = getllm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "Summerize this portion of the transcript."),
        ("human", "Summarize the following transcript: {text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser() 

    chunks = split_transcript(transcript)

    chuck_summeries = [map_chain.invoke( {"text" : chunk } ) for chunk in chunks]

    combined = "\n\n".join(chuck_summeries)

    # This function creates a chat prompt template for combining summaries into a single coherent summary.
    combine_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "You are a helpful assistant that combines summaries into a single coherent summary."
        "Into the final professional summary in bullet points format"),
        ("human", "{text}"),
        ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combine_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke( combined)

#This function generates a title for the transcript using the Mistral AI model.
def generate_title(transcript: str) -> str:
    "Generate a title for the transcript using the Mistral AI model."
    llm = getllm()
    
    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | ChatPromptTemplate.from_messages(
            [
                ("system", "Based on the content of the transcript, generate a concise and informative title."
                 "(max 8 words). Only retun the title, no other text.",
                 ),
                 ("human", "{text}"),
            ]
        ) 
        | llm | StrOutputParser()
    )

    return title_chain.invoke( transcript [:2000] )

 