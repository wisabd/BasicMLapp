import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from io import StringIO

st.cache_data.clear()
st.title('🏢 Real Estate Price Predictor')



st.write('In the area of Buenos Aires, Argentina. A Linear Regression (Machine Learning) powered app.')

url = "https://raw.githubusercontent.com/wisabd/BasicMLapp/master/buenos-aires-real-estate-1.csv"
# Fetch the CSV file content
response = requests.get(url)

if response.status_code == 200:
    # Read the CSV content into a DataFrame
    with st.expander('Dataset (Source: Private)'):
        st.write('**Raw Data**')
        data = pd.read_csv(StringIO(response.text))
        data

  

   2
    st.write('**X**')
        
else:
    st.write('In the notg;')
    print("Failed to fetch the CSV file.")
