import streamlit as st
import pandas as pd

st.title('🏢 Real Estate Price Predictor')

st.write('In the area of Buenos Aires, Argentina. A Linear Regression (Machine Learning) powered app.')
df = pd.read_csv("https://github.com/wisabd/BasicMLapp/blob/master/buenos-aires-real-estate-1.csv")
df
