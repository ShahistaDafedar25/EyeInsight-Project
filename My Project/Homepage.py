import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
import pandas as pd
import os

# Load the pre-trained model
model = tf.keras.models.load_model('C:\\Users\\Lenovo\\Documents\\Eye Disease Project\\models\\EyeInsight.keras')

# Preprocess the image for model input
def preprocess_image(image):
    resized_image = tf.image.resize(image, (256, 256))
    return resized_image

# Set page configuration
st.set_page_config(page_title="EyeInsight", page_icon=":eye:", layout="wide")

# Custom CSS for margin adjustments
st.markdown(
    """
    <style>
    .css-18e3th9 {
        padding-top: 0rem;
        padding-bottom: 10rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .css-1d391kg {
        padding-top: 0rem;
        padding-right: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# CSS for the horizontal countdown timer with a loading bar
countdown_css = """
<style>
#countdown {
    position: relative;
    width: 100%;
    height: 30px;
    margin: 20px 0;
    border: 1px solid #ccc;
    border-radius: 5px;
    overflow: hidden;
}

#loading-bar {
    position: absolute;
    width: 0;
    height: 100%;
    background-color: #3498db;
    animation: fill 10s linear forwards;
}

@keyframes fill {
    0% { width: 0; }
    100% { width: 100%; }
}
</style>
"""

# Sidebar menu configuration
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Prediction", "About Us"],
        icons=["house", "building-fill-gear", "envelope"],
        default_index=0
    )

# Home page
if selected == "Home":
    st.title("EyeInsight: Deep Learning Ocular Disease Prediction")
    st.header("Home Page",divider="rainbow")
    st.markdown("EyeInsight is a cutting-edge AI-driven platform dedicated to revolutionizing eye disease diagnosis. Utilizing advanced deep learning algorithms, our technology analyzes medical images to accurately detect and classify various ocular conditions. Designed to support healthcare professionals, EyeInsight aims to enhance diagnostic accuracy, streamline workflows, and ultimately improve patient outcomes. Join us in transforming the future of eye care with innovative, reliable, and accessible diagnostic solutions.")
    st.markdown(
        """
        <div style='display: flex; align-items: center;'>
            <div style='background:#FFA500; padding: 10px 20px; border-radius: 15px; display: inline-block;'>
                <span style='margin-right: 10px;'>Start Journey...........</span>
                <span>&#128640;</span> <!-- Rocket icon -->
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Prediction page
if selected == "Prediction":
    st.title("EyeInsight: Deep Learning Ocular Disease Prediction")
    st.header("Disease Prediction Page",divider="rainbow")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Please upload a clear image of the eye for accurate diagnosis.")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    captured_image = None
    with col2:
        st.subheader("Please stay steady for accurate diagnosis.")
        if st.button("Capture Image in 10 Seconds"):
            st.write("Get ready! Capturing image in 10 seconds...")

            # Display the countdown timer with loading bar
            st.markdown(countdown_css, unsafe_allow_html=True)
            countdown_placeholder = st.empty()
            image_placeholder = st.empty()

            for i in range(10, 0, -1):
                countdown_html = f"""
                <div id="countdown">
                    <div id="loading-bar" style="width: {10*(10-i)}%;"></div>
                </div>
                """
                countdown_placeholder.markdown(countdown_html, unsafe_allow_html=True)
                time.sleep(1)

            # Clear the countdown timer
            countdown_placeholder.empty()

            # Capture the image from the webcam
            capture = cv2.VideoCapture(0)
            ret, frame = capture.read()

            if ret:
                # Resize and convert the captured image
                resized_frame = cv2.resize(frame, (128, 128))
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                captured_image = img  # Save the captured image for analysis
                # Display the captured image
                image_placeholder.image(img, caption="Captured Image", width=128, use_column_width=True)
                # Save the captured image
                img.save("captured_image.jpg")
                st.success("Image captured and saved successfully!")
            else:
                st.error("Failed to capture image")

            # Release the webcam
            capture.release()

    # Analyze either the uploaded image or the captured image
    if uploaded_file is not None or captured_image is not None:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            image = captured_image

        resized_image = preprocess_image(image)
        resized_image_array = resized_image.numpy()
        normalized_image = resized_image_array / 255.0  # Normalize to [0.0, 1.0]
        st.image(normalized_image, caption='Image for Analysis', width=256)

        st.header("Predicting...", divider='grey')

        # Make predictions using the model
        predictions = model.predict(np.expand_dims(normalized_image, 0))

        # Define the classes
        classes = ['Bulging Eye', 'Cataract Eye', 'Cellulitis Eye', 'Crossed Eye', 'Normal Eye', 'Uveitis Eye', 'Keratoconus Eye']
        predicted_class = classes[np.argmax(predictions)]

        st.write(f"Predicted Disease: {predicted_class}")

        st.write("Disease Probabilities:")
        fig, ax = plt.subplots(figsize=(4, 4))
        colors = plt.cm.tab20.colors  # Use a custom color map
        wedges, texts, autotexts = ax.pie(predictions.flatten(), labels=classes, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.setp(texts, size=4)
        plt.setp(autotexts, size=8, weight="bold")  # Set the size and weight of the percentage labels
        plt.legend(wedges, classes, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))  # Move legend outside the pie chart
        st.pyplot(fig)

        st.header("For recommending doctors, click on 'See Doctors'", divider="grey")

        if "see_doctors" not in st.session_state:
            st.session_state.see_doctors = False

        if st.button("See Doctors") or st.session_state.see_doctors:
            st.session_state.see_doctors = True
            st.subheader("Doctors Data")

            # Change the working directory to the specified path
            os.chdir(r"C:\Users\Lenovo\Documents\Eye Disease Project\My Project")

            # Load the CSV data into a pandas DataFrame without using the first column as an index
            df = pd.read_csv("sample-data-Ophthalmologists.csv", encoding="ISO_8859-1", index_col=None)

            # Strip any leading or trailing spaces in the column names
            df.columns = df.columns.str.strip()

            col1, col2 = st.columns((2))

            with col1:
                # Multi-select widget for selecting states
                state = st.multiselect("Pick Your State", df["state"].unique())

            # Filter the DataFrame based on the selected state(s)
            if state:
                df2 = df[df["state"].isin(state)]
            else:
                df2 = df

            with col2:
                # Multi-select widget for selecting cities based on the filtered DataFrame
                city = st.multiselect("Pick Your City", df2["city"].unique())

            # Further filter the DataFrame based on the selected city(ies)
            if city:
                df3 = df2[df2["city"].isin(city)]
            else:
                df3 = df2

            # Specify the columns you want to display
            columns_to_display = ["Hospital Name", "Address", "Phone No.", "Email Address", "Website Url", "Star", "Rating", "Category Name"]

            # Display each row in the filtered DataFrame vertically with specified columns
            if not df3.empty:
                for index, row in df3.iterrows():
                    st.write("--------")
                    st.markdown('<div style="padding: 20px; margin-bottom: 10px; border-radius: 5px;">', unsafe_allow_html=True)
                    for column in columns_to_display:
                        if column in row:
                            st.markdown(f'''
                                <p style="background-color:#52D3D8; margin: 0; font-size: 25px; color: white;">
                                    <strong>{column}:</strong> {row[column]}
                                </p>
                            ''', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

# About Us page
if selected == "About Us":
    st.title("EyeInsight : Deep Learning Ocular Disease Prediction")
    st.header("About Us", divider='rainbow')
    
    # Brief information about the project and model
    st.markdown("""
        **Project Overview:**
        
        EyeInsight is an AI-powered platform that leverages deep learning to diagnose ocular diseases. By analyzing medical images, our model can accurately detect and classify various eye conditions, providing valuable assistance to healthcare professionals in making informed decisions. This technology aims to improve diagnostic accuracy, streamline clinical workflows, and enhance patient outcomes.
        
        **Model Details:**
        
        The EyeInsight model is built using a Convolutional Neural Network (CNN) architecture, which is trained on a diverse dataset of eye images. The model can identify conditions such as Bulging Eye, Cataract Eye, Cellulitis Eye, Crossed Eye, Normal Eye, Uveitis Eye, and Keratoconus Eye with high precision. Our goal is to support ophthalmologists with a reliable tool for early detection and treatment planning.
    """)

    # Feedback form
    st.subheader("We Value Your Feedback")
    email = st.text_input("Your Email Address")
    feedback = st.text_area("Your Feedback")

    if st.button("Submit Feedback"):
        if email and feedback:
            # Save feedback to a text file
            with open("feedback.txt", "a") as file:
                file.write(f"Email: {email}\nFeedback: {feedback}\n{'-'*40}\n")
            st.success("Thank you for your feedback!")
        else:
            st.error("Please provide both your email address and feedback.")
