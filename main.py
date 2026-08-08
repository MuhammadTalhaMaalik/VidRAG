from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_audio_chunks
from core.summerize import summarize_transcript, generate_title
from core.extractor import extract_actionable_items, extract_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

def run_pipeline(source: str) ->dict:
    print("Starting AI video Assitent")

    chunks = process_input(source)

    transcript = transcribe_audio_chunks(chunks)

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


if __name__ == "__main__":
  #CLI Entry point
 source = input("Enter Youtube URL or Local File Path: ").strip()
 #language = input("Language English: ").strip() or "english"
 result = run_pipeline(source)

 print("\n" + "=" * 60)
 print(f"Title: {result['Title']}")
 print(f"Summary: {result['Summary']}")
 print(f"Action_Items: {result['Action_Items']}")
 print(f"Key_Decisions: {result['Key_Decision']}")
 print(f"Open_Question: {result['Open_Question']}")
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
    


