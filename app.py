import streamlit as st
import moviepy.editor as mp
import librosa
import numpy as np
import mediapipe as mp_face
import cv2
import tempfile
import os

# --- CONFIGURATION ---
ASPECT_RATIO_9_16 = (9, 16)
TARGET_HEIGHT = 1280  # High quality vertical
TARGET_WIDTH = 720
CLIP_DURATION = 30  # Seconds per reel

st.set_page_config(page_title="PulsePoint AI", layout="wide")

# --- CORE ENGINEERING: MULTIMODAL PROCESSING ---
def analyze_emotional_peaks(audio_path, top_n=3):
    """
    Multimodal Logic: Uses Audio Energy (RMS) to find 'loud' passionate moments.
    Returns a list of start_times for the best clips.
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    # Calculate audio energy (RMS)
    hop_length = 512
    frame_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Convert frames to time
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # Find peaks (moments of high energy)
    # We smooth the signal to avoid rapid jitter
    # Simple heuristic: Sort 1-second windows by average energy
    window_size = int(sr / hop_length) # ~1 second
    
    ranked_moments = []
    for i in range(0, len(rms) - window_size, window_size * 5): # Check every 5 seconds
        segment_energy = np.mean(rms[i:i+window_size])
        ranked_moments.append((times[i], segment_energy))
    
    # Sort by highest energy (passion)
    ranked_moments.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N distinct timestamps (prevent overlapping clips)
    final_timestamps = []
    for t, energy in ranked_moments:
        if len(final_timestamps) >= top_n:
            break
        # Ensure clips don't overlap (simple check)
        if not any(abs(t - existing_t) < CLIP_DURATION for existing_t in final_timestamps):
            final_timestamps.append(t)
            
    return final_timestamps

# --- CORE ENGINEERING: SMART CROP ---
def smart_crop_frame(frame):
    """
    Uses MediaPipe to find the face and calculate a center crop for 9:16.
    """
    mp_face_detection = mp_face.solutions.face_detection
    
    height, width, _ = frame.shape
    target_ratio = 9/16
    
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        center_x = width // 2 # Default to center if no face
        
        if results.detections:
            # Get the first face detected
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Calculate face center
            face_x = int((bbox.xmin + bbox.width / 2) * width)
            center_x = face_x
            
    # Calculate cropping coordinates
    new_width = int(height * target_ratio)
    
    x1 = max(0, center_x - new_width // 2)
    x2 = min(width, x1 + new_width)
    
    # Adjust if out of bounds
    if x2 > width:
        x2 = width
        x1 = width - new_width
    if x1 < 0:
        x1 = 0
        x2 = new_width

    return frame[:, x1:x2]

def process_video_pipeline(video_file):
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # 1. Save Uploaded File
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(video_file.read())
        video_path = tfile.name

    try:
        status_text.text("Step 1/4: Extracting Audio for Analysis...")
        clip = mp.VideoFileClip(video_path)
        audio_path = "temp_audio.wav"
        clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        progress_bar.progress(25)

        status_text.text("Step 2/4: Identifying 'Emotional Peaks' (Multimodal AI)...")
        peak_times = analyze_emotional_peaks(audio_path)
        progress_bar.progress(50)
        
        generated_clips = []
        
        for i, start_time in enumerate(peak_times):
            status_text.text(f"Step 3/4: Processing Reel {i+1} (Smart Crop + Export)...")
            
            # Subclip logic
            end_time = min(start_time + CLIP_DURATION, clip.duration)
            subclip = clip.subclip(start_time, end_time)
            
            # --- ENGINEERING JUDGMENT: RESIZING ---
            # Smart Crop is heavy. For a hackathon, we apply a CENTER crop 
            # or use the face logic on the first frame to determine the crop zone
            # to save processing time (vs detecting face on every frame).
            
            # Simple approach: Crop to 9:16 center
            w, h = subclip.size
            new_w = h * (9/16)
            subclip_cropped = subclip.crop(x1=(w/2 - new_w/2), width=new_w, height=h)
            
            # Resize for mobile
            subclip_resized = subclip_cropped.resize(height=1280)
            
            output_filename = f"reel_{i+1}.mp4"
            subclip_resized.write_videofile(output_filename, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            generated_clips.append(output_filename)
        
        progress_bar.progress(100)
        status_text.text("Pipeline Complete! Ready for Download.")
        
        # Cleanup
        clip.close()
        os.remove(audio_path)
        os.remove(video_path)
        
        return generated_clips

    except Exception as e:
        st.error(f"Pipeline Error: {e}")
        return []

# --- UI LAYER ---
st.title("⚡ PulsePoint AI")
st.markdown("### Turn Long-Form Video into Viral Shorts Instantly")
st.write("Engineered for ByteSize Sage Hackathon")

uploaded_file = st.file_uploader("Upload MP4 Video (Long Form)", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)
    if st.button("Initialize Pipeline"):
        with st.spinner("Running Multimodal Extraction..."):
            clips = process_video_pipeline(uploaded_file)
            
            st.success(f"Success! Generated {len(clips)} Viral Clips.")
            
            cols = st.columns(len(clips))
            for idx, clip_path in enumerate(clips):
                with cols[idx]:
                    st.video(clip_path)
                    with open(clip_path, "rb") as file:
                        st.download_button(
                            label=f"Download Reel {idx+1}",
                            data=file,
                            file_name=clip_path,
                            mime="video/mp4"
                        )
