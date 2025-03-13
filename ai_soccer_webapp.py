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

def generate_detailed_ai_feedback(ai_ratings):
    feedback = {}
    for skill, rating in ai_ratings.items():
        if skill == "Dribbling":
            if rating >= 7:
                feedback[skill] = ("✅ Excellent ball control! Keep refining speed.", (0, 255, 0))
            elif rating >= 5:
                feedback[skill] = ("⚠️ Decent dribbling, but improve control at higher speeds.", (255, 255, 0))
            else:
                feedback[skill] = ("❌ Struggles with dribbling. Focus on close ball control.", (0, 0, 255))
        
        elif skill == "Passing":
            if rating >= 7:
                feedback[skill] = ("✅ Strong passing accuracy. Try increasing pass speed.", (0, 255, 0))
            elif rating >= 5:
                feedback[skill] = ("⚠️ Moderate passing. Work on consistency under pressure.", (255, 255, 0))
            else:
                feedback[skill] = ("❌ Passing needs work. Focus on target accuracy.", (0, 0, 255))

        elif skill == "Shooting":
            if rating >= 7:
                feedback[skill] = ("✅ Great shot power! Try improving shot placement.", (0, 255, 0))
            elif rating >= 5:
                feedback[skill] = ("⚠️ Decent shooting. Work on shot angles.", (255, 255, 0))
            else:
                feedback[skill] = ("❌ Shooting needs improvement. Focus on follow-through.", (0, 0, 255))

        elif skill == "Speed":
            if rating >= 7:
                feedback[skill] = ("✅ Fast sprint speed! Work on endurance.", (0, 255, 0))
            elif rating >= 5:
                feedback[skill] = ("⚠️ Average speed. Try explosive sprint drills.", (255, 255, 0))
            else:
                feedback[skill] = ("❌ Speed is low. Focus on acceleration training.", (0, 0, 255))

        elif skill == "Agility":
            if rating >= 7:
                feedback[skill] = ("✅ Quick footwork! Maintain consistency in lateral moves.", (0, 255, 0))
            elif rating >= 5:
                feedback[skill] = ("⚠️ Agility is decent. Improve quick direction changes.", (255, 255, 0))
            else:
                feedback[skill] = ("❌ Agility needs improvement. Do ladder and cone drills.", (0, 0, 255))

    return feedback


# Function to generate AI feedback
def generate_ai_feedback(ai_ratings):
    feedback = {}
    for skill, rating in ai_ratings.items():
        if rating >= 7:
            feedback[skill] = ("✅ Strong performance in " + skill, (0, 255, 0))  # Green for positive
        elif rating >= 5:
            feedback[skill] = ("⚠️ Decent " + skill + ", but can be improved.", (255, 255, 0))  # Yellow for neutral
        else:
            feedback[skill] = ("❌ Needs improvement in " + skill + ". Focus on drills.", (0, 0, 255))  # Red for improvement
    return feedback

# Function to determine AI-recommended drills
def get_ai_recommended_drills(ai_ratings):
    weakest_skill = min(ai_ratings, key=ai_ratings.get)  # Find the lowest-rated skill
    _, drills = analyze_skills(
        ai_ratings["Dribbling"], ai_ratings["Passing"], ai_ratings["Shooting"], ai_ratings["Speed"], ai_ratings["Agility"]
    )
    return weakest_skill, drills

# Function to process the video and overlay AI feedback
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
    ai_feedback = generate_detailed_ai_feedback(ai_ratings)  # AI generates detailed feedback
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

        # Keep detailed AI feedback text on screen at all times with smaller font
        y_offset = 50
        font_scale = 0.6  # Reduce font size
        font_thickness = 1  # Reduce thickness for smaller text
        for skill, (text, color) in ai_feedback.items():
            cv2.rectangle(frame, (30, y_offset - 20), (800, y_offset + 10), (0, 0, 0), -1)  # Background for readability
            cv2.putText(frame, text, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)
            y_offset += 30  # Reduce spacing for compact display

        out.write(frame)

    cap.release()
    out.release()

    return temp_output_path, ai_ratings, ai_feedback, weakest_skill, recommended_drills


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
                st.write(f"**{skill}:** {rating}/10 - {ai_feedback[skill][0]}")

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
