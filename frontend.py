import streamlit as st
import pandas as pd
import requests
import io

# FastAPI endpoint URLs
PREDICT_URL = "https://starsizeapp-1.onrender.com/predict/"
PLOT_URL = "https://starsizeapp-1.onrender.com/plot/"

st.title("Star Size Prediction App")

# Instructions for uploading the CSV file
st.write("""### Instructions for Uploading CSV File
Please upload a CSV file that contains columns in the following order:
- **Column 1**: Brightness of the stars (numeric values)
- **Column 2**: Size of the stars (numeric values)
""")

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