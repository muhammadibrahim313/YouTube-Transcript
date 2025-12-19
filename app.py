import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
from dotenv import load_dotenv
import os
import yt_dlp

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
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "list=" in url:
            st.warning("Playlists are not supported. Please provide a single video link.")
            return None
    else:
        raise ValueError("Invalid YouTube URL")

# Function to download audio using yt-dlp
def download_audio(video_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'audio.%(ext)s',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return 'audio.mp3'

# Function to get transcript (with fallback to Whisper)
def get_transcript(video_id, languages=['en'], fallback=False, video_url=None):
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=languages)
        full_text = " ".join([entry['text'] for entry in transcript])
        return full_text, transcript  # Return both plain text and raw for formatted display
    except Exception as e:
        if fallback and video_url:
            st.info("No subtitles available. Falling back to audio transcription using Groq Whisper...")
            audio_file = download_audio(video_url)
            with open(audio_file, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=file,
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            full_text = transcription.text
            raw_transcript = transcription.segments  # List of {'start': float, 'end': float, 'text': str}
            os.remove(audio_file)  # Clean up
            return full_text, raw_transcript
        else:
            raise ValueError(f"Could not fetch transcript: {str(e)}")

# Function to format transcript with timestamps
def format_transcript(raw_transcript):
    formatted = ""
    for entry in raw_transcript:
        if 'start' in entry and 'end' in entry:  # For Whisper format
            start_time = f"{int(entry['start'] // 60):02d}:{int(entry['start'] % 60):02d}"
            end_time = f"{int(entry['end'] // 60):02d}:{int(entry['end'] % 60):02d}"
            formatted += f"[{start_time} - {end_time}] {entry['text']}\n"
        else:  # For YouTube format
            start_time = f"{int(entry['start'] // 60):02d}:{int(entry['start'] % 60):02d}"
            formatted += f"[{start_time}] {entry['text']}\n"
    return formatted

# Function to generate summary using Groq with streaming
def generate_summary(transcript, custom_prompt="Summarize the following transcript:"):
    content = f"{custom_prompt} {transcript}"
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",  # As per your specified model; adjust if it's not available on Groq
        messages=[
            {
                "role": "user",
                "content": content
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
        summary_placeholder.markdown(summary + " ▌")  # Add cursor for streaming effect
    summary_placeholder.markdown(summary)  # Final update without cursor
    return summary

# Streamlit UI - Further improved with better layout, colors, and options
st.set_page_config(page_title="YouTube Transcript Summarizer", page_icon="🎥", layout="wide")

st.title("🎥 YouTube Transcript Summarizer")
st.markdown("""
This app fetches the transcript from a YouTube video (using subtitles if available, or audio transcription as fallback) and generates a real-time AI summary using Groq. 
You can choose to generate a summary or just view the transcript. The complete transcript is always shown in a formatted way with timestamps!
""")

# Sidebar for options
with st.sidebar:
    st.header("Settings")
    languages = st.multiselect("Preferred Languages", options=['en', 'de', 'fr', 'es', 'it', 'ja', 'ko', 'zh', 'other'], default=['en'],
                               help="Select languages in priority order. Defaults to English.")
    custom_prompt = st.text_input("Custom Summary Prompt", value="Summarize the following transcript:",
                                  help="Customize the prompt sent to the AI for summarization, e.g., 'Summarize key points from:'")
    generate_summary_option = st.checkbox("Generate AI Summary", value=True, help="Uncheck if you only want the transcript.")
    fallback_transcription = st.checkbox("Fallback to Audio Transcription (Groq Whisper)", value=True, 
                                         help="If no subtitles are available, automatically transcribe the audio. Requires yt-dlp and may take time.")
    st.markdown("---")
    st.caption("Powered by YouTube Transcript API, yt-dlp, Groq AI & Whisper")
    st.caption("Note: If the model 'openai/gpt-oss-120b' is unavailable, update it in app.py to a valid Groq model like 'llama3-70b-8192'.")
    st.caption("Ensure ffmpeg is installed for audio processing.")

# Main content
col1, col2 = st.columns([3, 1])
with col1:
    video_url = st.text_input("YouTube Video Link", placeholder="https://youtu.be/VIDEO_ID or https://www.youtube.com/watch?v=VIDEO_ID")

if st.button("Process Video", use_container_width=True):
    if video_url:
        with st.spinner("Extracting video ID and fetching transcript..."):
            try:
                video_id = get_video_id(video_url)
                if not video_id:
                    st.stop()
                transcript_text, raw_transcript = get_transcript(video_id, languages if languages else ['en'], 
                                                                fallback=fallback_transcription, video_url=video_url)
                st.success("Transcript obtained successfully!")
                
                # Always display formatted transcript in a good way
                st.subheader("Complete Transcript (with Timestamps)")
                formatted_transcript = format_transcript(raw_transcript)
                st.text_area("Formatted Transcript", formatted_transcript, height=400)
                
                # Option to download transcript
                st.download_button(
                    label="Download Transcript as TXT",
                    data=formatted_transcript,
                    file_name="transcript.txt",
                    mime="text/plain"
                )
                
                # Generate summary if option is selected
                if generate_summary_option:
                    st.subheader("Generating Summary...")
                    summary = generate_summary(transcript_text, custom_prompt)
                    
                    st.subheader("AI-Generated Summary")
                    st.markdown(summary)
                    
                    # Add copy button for summary
                    st.button("Copy Summary", on_click=lambda: st.session_state.update({"summary": summary}))
                    if "summary" in st.session_state:
                        st.code(st.session_state["summary"], language="text")
            except ValueError as ve:
                st.error(str(ve))
                st.info("Tip: Some videos may not have transcripts available. Enable 'Fallback to Audio Transcription' or try a different video.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                st.info("If the error persists, ensure the video has public access and try again.")
    else:
        st.warning("Please enter a valid YouTube video link.")
