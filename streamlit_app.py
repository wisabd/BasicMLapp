import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from io import StringIO


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

  

    maskcp = data["place_with_parent_names"].str.contains("Capital Federal")
    maskpt = data["property_type"] == "apartment"
    maskpr = data["price"] < 400_000

    data = data[maskcp & maskpt & maskpr]
    low, high = data["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask = data["surface_covered_in_m2"].between(low, high)
    df = data[mask]

    st.write('matplotlib.__version__')
    
    st.pyplot.figure(figsize=(10, 6))  # Set the figure size
    st.pyplot.hist(df['surface_covered_in_m2'], bins=30, color='skyblue', edgecolor='black')

    # Add labels and title
    st.pyplot.xlabel('Area [sq meters]', fontsize=12)
    st.pyplot.ylabel('Frequency', fontsize=12)
    st.pyplot.title('Distribution of Apartment Sizes', fontsize=14)

    # Show the plot
    st.pyplot.show()
        
    st.write('**X**')
        
else:
    st.write('In the notg;')
    print("Failed to fetch the CSV file.")
