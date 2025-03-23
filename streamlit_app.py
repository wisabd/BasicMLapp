import streamlit as st

st.title('🏢 Real Estate Price Predictor')

st.write('In the area of Buenos Aires, Argentina. A Linear Regression (Machine Learning) powered app.')
df = pd.read_csv("https://vm.wqu.edu/lab/tree/work/ds-curriculum/020-housing-in-buenos-aires/data/buenos-aires-real-estate-1.csv")
df
