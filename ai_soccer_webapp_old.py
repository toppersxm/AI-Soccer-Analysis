import random
import cv2
import mediapipe as mp
import streamlit as st
import tempfile
import os
import numpy as np

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

def generate_feedback(landmarks):
    feedback = []
    if landmarks:
        left_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
        left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        
        # Basic posture analysis
        if left_knee.y < left_hip.y and right_knee.y < right_hip.y:
            feedback.append(("Good posture! Keep your knees bent slightly for better balance.", (0, 255, 0)))
        else:
            feedback.append(("Try bending your knees slightly more for improved stability.", (0, 0, 255)))
        
        # Passing technique
        if abs(left_hip.x - right_hip.x) < 0.1:
            feedback.append(("Great hip alignment! Keep following through with your passing foot.", (0, 255, 0)))
        else:
            feedback.append(("Align your hips properly for a more accurate pass.", (0, 0, 255)))
        
        # Shooting stance
        if left_knee.y > left_hip.y or right_knee.y > right_hip.y:
            feedback.append(("Shift your weight forward for more power in your shot.", (0, 0, 255)))
    
    return feedback

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
    feedback_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            feedback = generate_feedback(results.pose_landmarks.landmark)
            feedback_list.extend([fb[0] for fb in feedback])
            
            for i, (fb_text, fb_color) in enumerate(feedback):
                cv2.putText(frame, fb_text, (50, 50 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, fb_color, 3, cv2.LINE_AA)
            
            for landmark in results.pose_landmarks.landmark:
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

        out.write(frame)

    cap.release()
    out.release()
    
    return temp_output_path, feedback_list

def main():
    st.title("AI Soccer Drill & Video Analysis")
    
    st.header("Self-Assessment: Rate Your Skills (1-10)")
    dribbling = st.slider("Dribbling", 1, 10, 5)
    passing = st.slider("Passing", 1, 10, 5)
    shooting = st.slider("Shooting", 1, 10, 5)
    speed = st.slider("Speed", 1, 10, 5)
    agility = st.slider("Agility", 1, 10, 5)
    
    if st.button("Analyze Skills & Get Drills"):
        weakest_skill, drills = analyze_skills(dribbling, passing, shooting, speed, agility)
        st.write(f"Your weakest skill is: **{weakest_skill}**")
        st.write("### Recommended Drills:")
        for drill in drills:
            st.write(f"- {drill}")
    
    st.header("Upload Soccer Training Video for AI Analysis")
    uploaded_video = st.file_uploader("Upload a soccer video", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_video.read())
            temp_video_path = temp_file.name

        st.video(temp_video_path, format='video/mp4')
        st.write("Processing video... This may take a moment.")
        output_video_path, feedback = process_video(temp_video_path)
        
        if os.path.exists(output_video_path):
            with open(output_video_path, "rb") as video_file:
                st.download_button(
                    label="Download Processed Video",
                    data=video_file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
            
            st.write("### Processed Video with AI Analysis:")
            st.video(output_video_path, format='video/mp4')
            
            if feedback:
                st.write("### Movement Feedback:")
                for comment in set(feedback):
                    st.write(f"- {comment}")
        else:
            st.error("Error processing video. Please try again.")

if __name__ == "__main__":
    main()

