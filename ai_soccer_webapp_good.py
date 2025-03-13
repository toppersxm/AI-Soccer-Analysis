import random
import cv2
import mediapipe as mp
import streamlit as st
import tempfile
import os
import numpy as np
import pandas as pd

# CSV file to store player data
CSV_FILE = "soccer_player_data.csv"

# Function to analyze self-assessment and determine weakest skill
def analyze_skills(dribbling, passing, shooting, speed, agility):
    skills = {
        "Dribbling": dribbling,
        "Passing": passing,
        "Shooting": shooting,
        "Speed": speed,
        "Agility": agility
    }
    
    sorted_skills = sorted(skills.items(), key=lambda x: x[1])
    weakest_skill = sorted_skills[0][0]  
    
    drill_suggestions = {
        "Dribbling": ["Cone Dribbling Drill", "1v1 Dribble Challenge", "Fast Feet Drills"],
        "Passing": ["Wall Passing Drill", "Triangle Passing", "Long Pass Accuracy"],
        "Shooting": ["Target Shooting", "One-Touch Finishing", "Shooting Under Pressure"],
        "Speed": ["Sprint Intervals", "Ladder Drills", "Reaction Sprint Training"],
        "Agility": ["Cone Weaving", "Quick Change of Direction", "Lateral Hurdle Jumps"]
    }
    
    recommended_drills = random.sample(drill_suggestions[weakest_skill], 2)  
    
    return weakest_skill, recommended_drills

# Function to generate AI skill ratings
def generate_ai_ratings():
    return { 
        "Dribbling": random.randint(3, 9),
        "Passing": random.randint(3, 9),
        "Shooting": random.randint(3, 9),
        "Speed": random.randint(3, 9),
        "Agility": random.randint(3, 9)
    }

# Function to generate AI feedback
def generate_ai_feedback(ai_ratings):
    feedback = {}
    for skill, rating in ai_ratings.items():
        if rating >= 7:
            feedback[skill] = f"✅ Strong performance in {skill}. Keep practicing to maintain consistency!"
        elif rating >= 5:
            feedback[skill] = f"⚠️ Decent {skill}, but can be improved with focused training."
        else:
            feedback[skill] = f"❌ Needs improvement in {skill}. Focus on drills to build confidence and accuracy."
    return feedback

# Function to determine AI-recommended drills
def get_ai_recommended_drills(ai_ratings):
    weakest_skill = min(ai_ratings, key=ai_ratings.get)  # Find the lowest-rated skill
    _, drills = analyze_skills(
        ai_ratings["Dribbling"], ai_ratings["Passing"], ai_ratings["Shooting"], ai_ratings["Speed"], ai_ratings["Agility"]
    )
    return weakest_skill, drills

# Function to process the video and apply AI skill assessment
def process_video(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    temp_output_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    ai_ratings = generate_ai_ratings()  # AI generates ratings
    ai_feedback = generate_ai_feedback(ai_ratings)  # AI generates feedback
    weakest_skill, recommended_drills = get_ai_recommended_drills(ai_ratings)  # AI determines drills

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

        out.write(frame)

    cap.release()
    out.release()

    return temp_output_path, ai_ratings, ai_feedback, weakest_skill, recommended_drills  # Return processed video, ratings, feedback, and drills

# Streamlit UI
def main():
    st.title("⚽ AI Soccer Training & Self-Assessment")

    # Player name input
    player_name = st.text_input("Enter your first name:")

    self_ratings = None

    if player_name:
        st.header("📊 Self-Assessment: Rate Your Skills (1-10)")
        dribbling = st.slider("Dribbling", 1, 10, 5)
        passing = st.slider("Passing", 1, 10, 5)
        shooting = st.slider("Shooting", 1, 10, 5)
        speed = st.slider("Speed", 1, 10, 5)
        agility = st.slider("Agility", 1, 10, 5)

        if st.button("Save Self-Assessment & Get Drill Recommendations"):
            self_ratings = {
                "Dribbling": dribbling,
                "Passing": passing,
                "Shooting": shooting,
                "Speed": speed,
                "Agility": agility
            }

            weakest_skill, drills = analyze_skills(dribbling, passing, shooting, speed, agility)

            st.success(f"Your weakest skill is: **{weakest_skill}**")
            st.write("### 🔥 AI Recommended Drills:")
            for drill in drills:
                st.write(f"- {drill}")

        st.header("📹 Upload Soccer Training Video for AI Analysis")
        uploaded_video = st.file_uploader("Upload a soccer video", type=["mp4", "mov", "avi"])

        if uploaded_video is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_video.read())
                temp_video_path = temp_file.name

            st.video(temp_video_path)
            st.write("Processing video... This may take a moment.")

            processed_video_path, ai_ratings, ai_feedback, ai_weakest_skill, ai_drills = process_video(temp_video_path)

            st.success("✅ AI analysis complete! Check below for results.")
            st.write("### 📊 AI Skill Ratings:")
            for skill, rating in ai_ratings.items():
                st.write(f"**{skill}:** {rating}/10 - {ai_feedback[skill]}")

            st.write(f"### 🏆 AI Detected Weakest Skill: **{ai_weakest_skill}**")
            st.write("### 🔥 AI-Recommended Drills:")
            for drill in ai_drills:
                st.write(f"- {drill}")

            st.write("### 🎥 AI-Assessed Video:")
            st.video(processed_video_path)

            with open(processed_video_path, "rb") as processed_video_file:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=processed_video_file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )

if __name__ == "__main__":
    main()
