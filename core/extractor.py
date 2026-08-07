# Actionable items , decisions , Questions 

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter


def getllm():
    return ChatMistralAI(
        model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"), temperature = 0.2)
#This function is used to split the transcript into smaller segments so we cannot excceed the token limit of the model. But it not implemented in the current version of the code.
#def split_transcript(transcript: str) -> list:
    "Split the transcript into smaller segments for summarization."
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200,
    )
    return text_splitter.split_text(transcript)


def build_chain(system_prompt: str):
    "Build a chain for summarization tasks using the Mistral AI model."
    llm = getllm()

    return(
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}"),
        ]
    ) | llm | StrOutputParser()
    )

def extract_actionable_items(transcript: str) -> str:

    chain = build_chain(
        "You are an expert analyst. From the transcript,"
        "extract all action items."
        "Format as a numbered list with each item on a new line. If not found say 'No action items found.'"
    )     
    return chain.invoke(transcript)

def extract_decisions(transcript: str) -> str:

    chain = build_chain(
        "You are an expert analyst. From the transcript,"
        "extract all decisions."
        "Format as a numbered list with each item on a new line. If not found say 'No decisions found.'"
    )     
    return chain.invoke(transcript)

def extract_questions(transcript: str) -> str:

    chain = build_chain(
        "You are an expert analyst. From the transcript,"
        "extract all questions."
        "Format as a numbered list with each item on a new line. If not found say 'No questions found.'"
    )     
    return chain.invoke(transcript)


