from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware


# Create an instance of the FastAPI application
app = FastAPI()

# Allow streamlit to make calls to our API endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://star-size.streamlit.app/"],  # Replace with your Streamlit app URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Coefficients obtained from the linear regression training
W = 1.982015  # Weight (slope)
b = 9.500380  # Bias (intercept)

@app.get('/')
def default():
    """
    Default endpoint to check if the API is running.
    
    Returns:
        A dictionary indicating that the app is running.
    """
    return {'App': 'Running'}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to make predictions based on input data.

    Args:
        file (UploadFile): A CSV file containing 'inputs' and 'targets' columns.

    Returns:
        StreamingResponse: A CSV file with the original data plus predictions.
    """
    # Read the contents of the uploaded CSV file
    contents = await file.read()
    
    # Load the contents into a DataFrame
    df = pd.read_csv(io.BytesIO(contents))
    
    # Rename columns to a simpler format for ease of access
    df.columns = ['inputs', 'targets']

    # Calculate predictions using the linear regression model
    df['predictions'] = W * df['inputs'] + b

    # Convert the DataFrame with predictions back to CSV format
    output = df.to_csv(index=False).encode('utf-8')
    
    # Return the CSV as a downloadable file
    return StreamingResponse(io.BytesIO(output), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})

@app.post("/plot/")
async def plot(file: UploadFile = File(...)):
    """
    Endpoint to generate a plot of actual vs predicted values.

    Args:
        file (UploadFile): A CSV file containing 'inputs' and 'targets' columns.

    Returns:
        StreamingResponse: An image of the generated plot.
    """
    # Read the contents of the uploaded CSV file
    contents = await file.read()
    
    # Load the contents into a DataFrame
    df = pd.read_csv(io.BytesIO(contents))

    # Create a new figure for the plot
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of actual targets
    plt.scatter(df['inputs'], df['targets'], color='royalblue', label='Actual Targets',marker='x')

    # Calculate predictions for plotting the line of best fit
    df['predictions'] = W * df['inputs'] + b

    # RMSE calculation
    rmse_score = np.mean(np.square(df['predictions'].values - df['targets'].values))
    
    # Plot the line of best fit (predictions)
    plt.plot(df['inputs'], df['predictions'], color='k', label='Predictions', linewidth=2)

    # Set the title and labels for the plot
    plt.title(f'Linear Regression for Stars Data (RMSE:- {round(rmse_score, 1)})', color='maroon', weight='bold', fontsize=15)
    plt.xlabel('Brightness', color='m', weight='bold', fontsize=13)
    plt.ylabel('Size', color='m', weight='bold', fontsize=13)
    plt.legend()

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    
    # Seek to the beginning of the buffer to read its contents
    buf.seek(0)
    
    # Close the plot to free up memory
    plt.close()

    # Return the plot as a downloadable image
    return StreamingResponse(buf, media_type="image/png", headers={"Content-Disposition": "attachment; filename=plot.png"})