import streamlit as st
import pandas as pd
import requests
import io

# FastAPI endpoint URLs
PREDICT_URL = "https://starsizeapp-1.onrender.com/predict/"
PLOT_URL = "https://starsizeapp-1.onrender.com/plot/"

# Set page configuration for title and favicon
st.set_page_config(
    page_title="Star Size Predictor",  # Title that appears on the browser tab
    page_icon="⭐"  # Star emoji as favicon
)

# Add custom CSS to set a galaxy background image
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://4kwallpapers.com/images/walls/thumbs_3t/10307.jpg");
    background-size: cover;
    background-position: center;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Star Size Prediction App")

# Box for additional information or instructions
with st.container():
    st.markdown(
        """
        <div style="background-color: rgba(30, 30, 30, 0.9); padding: 10px; border-radius: 10px; color: royalblue;">
            <p>The aim of this app is to predict the sizes of stars based on their brightness.</p>
            <p></p>
        </div>
        <p></p>
        <div style="background-color: rgba(30, 30, 30, 0.9); padding: 20px; border-radius: 10px; color: white;">
            <h3>Instructions for Uploading CSV File</h3>
            <p>You can download the <a href="https://drive.google.com/uc?id=1Rp8JATmZGsTv-mlYz9KzTgYJDB4DlC5c" style="color: royalblue;">demo csv file</a> or create your own data using <a href="https://github.com/SpartificialUdemy/StarSizeApp/blob/main/data.py" style="color: royalblue;">this file</a>.</p>
            <p>Please upload a CSV file that contains columns in the following order:</p>
            <ul>
                <li><b>Column 1:</b> Brightness of the stars (numeric values)</li>
                <li><b>Column 2:</b> Size of the stars (numeric values)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write(" ")


# Step 1: Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Initialize session state to store predictions
if 'predictions_csv' not in st.session_state:
    st.session_state.predictions_csv = None
if 'predicted_data' not in st.session_state:
    st.session_state.predicted_data = None

if uploaded_file is not None:
    # Step 2: Read the CSV file into a DataFrame for display
    df = pd.read_csv(uploaded_file)

    # Reset the file cursor to the beginning for the POST request
    uploaded_file.seek(0)

    # Step 3: Predict Star Sizes automatically on file upload if not already done
    if st.session_state.predicted_data is None:
        with st.spinner("Generating predictions..."):
            # Send the original uploaded CSV file to the FastAPI predict endpoint
            response = requests.post(PREDICT_URL, files={"file": uploaded_file})

            if response.status_code == 200:
                # Store the predictions CSV in session state
                st.session_state.predictions_csv = io.BytesIO(response.content)
                st.session_state.predicted_data = pd.read_csv(st.session_state.predictions_csv)

                # Display the uploaded and predicted data side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Uploaded Data:")
                    st.dataframe(df)
                with col2:
                    st.subheader("Predicted Star Sizes:")
                    st.dataframe(st.session_state.predicted_data)
            else:
                st.error("Error occurred while predicting star sizes.")
    else:
        # If predictions already exist, just display them
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Data:")
            st.dataframe(df)
        with col2:
            st.subheader("Predicted Star Sizes:")
            st.dataframe(st.session_state.predicted_data)

# Step 4: Plot the graph
if st.session_state.predictions_csv is not None and st.session_state.predicted_data is not None:
    if st.button("Plot Linear Regression"):
        with st.spinner("Generating plot..."):
            # Use the existing predictions CSV file to generate the plot
            st.session_state.predictions_csv.seek(0)  # Reset cursor
            plot_response = requests.post(PLOT_URL, files={"file": st.session_state.predictions_csv})

            if plot_response.status_code == 200:
                st.image(plot_response.content, caption="Linear Regression Plot")
            else:
                st.error("Error occurred while generating the plot.")
