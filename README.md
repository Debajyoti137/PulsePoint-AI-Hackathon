⚡ PulsePoint AI
  ByteSize Sage AI Hackathon Submission

Live Preview: https://pulsepoint-ai-hackathon-8mexthds2gzrzsv6zzq3cq.streamlit.app/

🚀 The Pipeline
PulsePoint AI is an automated video processing pipeline designed to extract high-value "golden nuggets" from long-form content.

🛠️ Architecture
1.  Ingestion: Accepts MP4 files via Streamlit interface.
2.  Multimodal Analysis (The "Brain"): - Uses Librosa to extract audio energy (RMS).
    - Identifies "Emotional Peaks" by detecting sustained high-volume segments (indicative of passionate speaking).
3.  Visual Engineering:
    - MoviePy handles the heavy lifting of video slicing.
    - Automatically crops horizontal (16:9) video to vertical (9:16) for Reels/TikTok optimization.
4.  Delivery: Exports 3 top-ranked viral clips ready for upload.

 🔧 Tech Stack
- Frontend: Streamlit
- Video Processing: MoviePy, OpenCV
- Audio Analysis: Librosa
- Language: Python

 🏃 How to Run
1. `pip install -r requirements.txt`
2. `streamlit run app.py`
