import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
    
    fig, ax = plt.subplots()
    ax.hist(df['surface_covered_in_m2'], bins=30, color='skyblue', edgecolor='black')
    
    ax.set_title("Distribution of Apartment Sizes")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Area [sq meters]")

    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.scatter(df['surface_covered_in_m2'], df['price_aprox_usd'], c='blue', label='Data Points')
    ax.set_xlabel('Area [Sq metres]')
    ax.set_ylabel('Price [USD]')
    ax.set_title('Scatter Plot from DataFrame')
    st.pyplot(fig)

    X_train = df["surface_covered_in_m2"]
    target = "price_aprox_usd"
    y_train = df[target]
    y_mean = y_train.mean()
    y_pred_baseline = [y_mean] * len(y_train)
    baseline_df = pd.DataFrame({
    'surface_covered_in_m2': X_train,
    'baseline_prediction': y_pred_baseline})

    fig = go.Figure()

    # Add scatter plot for actual data
    fig.add_trace(
        go.Scatter(
        x=X_train,
        y=y_train,
        mode='markers',
        name='Actual Data Points',
        marker=dict(color='blue')
    ) ) 
    fig.add_trace(
    go.Scatter(
        x=X_train,
        y=y_pred_baseline,
        mode='lines',
        name='Baseline Prediction',
        line=dict(color='red')
    ))
    fig.update_layout(
    title="Buenos Aires: Price vs. Area",
    xaxis_title="Area [Sq metres]",
    yaxis_title="Price [USD]",
    legend_title="Legend"
    )

    st.plotly_chart(fig)

    url = "https://raw.githubusercontent.com/wisabd/BasicMLapp/master/buenos-aires-test-features.csv"
    # Fetch the CSV file content
    response = requests.get(url)
    if response.status_code == 200:
        X_test = pd.read_csv(StringIO(response.text))

    mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
    model = LinearRegression()
    X_train = df["surface_covered_in_m2"].values  # Convert to NumPy array
    X_train = X_train.reshape(-1, 1)
    model.fit(X_train, y_train)
    y_pred_training = model.predict(X_train)
    y_pred_training[:5]
    mae_training = mean_absolute_error(y_train, y_pred_training)
    features = ["surface_covered_in_m2"]
    X_test = X_test[features]
    y_pred_test = pd.Series(model.predict(X_test))
    intercept = round(model.intercept_,2)
    coefficient = model.coef_[0]

    st.write(f"apt_price = {intercept} + {coefficient} * surface_covered")



     fig = go.Figure()

    # Add scatter plot for actual data
    fig.add_trace(
        go.Scatter(
        x=X_train,
        y=y_train,
        mode='markers',
        name='Actual Data Points',
        marker=dict(color='blue')
    ) ) 
    fig.add_trace(
    go.Scatter(
        x=X_train,
        y=model.predict(X_train),
        mode='lines',
        name='Post-iteration Prediction : Linear Model',
        line=dict(color='red')
    ))
    fig.update_layout(
    title="Buenos Aires: Price vs. Area",
    xaxis_title="Area [Sq metres]",
    yaxis_title="Price [USD]",
    legend_title="Legend"
    )

    st.plotly_chart(fig)
    


else:
    st.write('In the notg;')
    print("Failed to fetch the CSV file.")
