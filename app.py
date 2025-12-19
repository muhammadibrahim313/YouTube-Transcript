import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file. Please add it and restart the app.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Function to extract video ID from URL
def get_video_id(url):
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    elif "youtube.com" in url:
        return url.split("v=")[1].split("&")[0]
    else:
        raise ValueError("Invalid YouTube URL")

# Function to get transcript
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        raise ValueError(f"Could not fetch transcript: {str(e)}")

# Function to generate summary using Groq
def generate_summary(transcript):
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",  # Use the model you specified; adjust if needed
        messages=[
            {
                "role": "user",
                "content": f"Summarize the following transcript: {transcript}"
            }
        ],
        temperature=1,
        max_completion_tokens=8192,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None
    )
    
    summary = ""
    summary_placeholder = st.empty()
    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        summary += content
        summary_placeholder.markdown(summary)  # Stream the output in real-time
    return summary

# Streamlit UI
st.title("YouTube Transcript Summarizer")

st.markdown("""
Enter a YouTube video link below to fetch its transcript and generate a real-time summary using Groq AI.
""")

video_url = st.text_input("YouTube Video Link", placeholder="https://youtu.be/VIDEO_ID")

if st.button("Generate Summary"):
    if video_url:
        with st.spinner("Fetching transcript..."):
            try:
                video_id = get_video_id(video_url)
                transcript = get_transcript(video_id)
                st.success("Transcript fetched successfully!")
                
                st.subheader("Generating Summary...")
                summary = generate_summary(transcript)
                
                st.subheader("Full Summary")
                st.markdown(summary)
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a valid YouTube link.")
