import streamlit as st
import pandas as pd
import requests
from io import StringIO


st.title('🏢 Real Estate Price Predictor')



st.write('In the area of Buenos Aires, Argentina. A Linear Regression (Machine Learning) powered app.')

url = "https://raw.githubusercontent.com/wisabd/BasicMLapp/blob/master/buenos-aires-real-estate-1.csv"
# Fetch the CSV file content
response = requests.get(url)
if response.status_code == 200:
    # Read the CSV content into a DataFrame
    data = pd.read_csv(StringIO(response.text))
    print(data)
else:
    print("Failed to fetch the CSV file.")
