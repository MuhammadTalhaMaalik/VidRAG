from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_audio_chunks
from core.summerize import summarize_transcript, generate_title
from core.extractor import extract_actionable_items, extract_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question
from os import name

load_dotenv()

def run_pipeline(source: str, language: str = "english") ->dict:
    print("Starting AI video Assitent")

    chunks = process_input(source)

    transcript = transcribe_audio_chunks(chunks, language = language)

    print(f"raw transcription (first 300 characters) {transcript[: 300]}")

    title  = generate_title(transcript)

    summary = summarize_transcript(transcript)

    action_items = extract_actionable_items(transcript)

    decision = extract_decisions(transcript)

    questions = extract_questions(transcript)

    rag_chain = build_rag_chain(transcript)

    return {
        "Title": title,
        "Transcript": transcript,
        "Summary": summary,
        "Action_Items": action_items,
        "Key_Decision": decision,
        "Open_Question": questions,
        "Rag_Chain": rag_chain,
    }


if name == "_main_":
 #CLI Entry point
 source = input("Enter Youtube URL or Local File Path: ").strip()
 language = input("Language English: ").strip() or "english"
 result = run_pipeline(source, language)

 print("\n" + "=" * 60)
 print(f"Title: {result['title']}")
 print(f"Summary: {result['summary']}")
 print(f"Action_Items: {result['action_atems']}")
 print(f"Key Decisions: {result['key_decision']}")
 print(f"Open Question: {result['open_question']}")
 print( "=" * 60)

 #Phase 2 : Chat with your meeting via RAG

 print("\n Chat with bot about video. (type 'exit' to quit)\n")
 rag_chain = result["rag_chain"] 
 while True:
    question = input("You: ").strip()
    if question.lower() in ["exit", "quit", "q"]:
       print("Goodbye!!")
       break
    if not question:
        continue
    answer = ask_question(rag_chain, question)

    print(f"\n Assistent: {answer}\n")
    


