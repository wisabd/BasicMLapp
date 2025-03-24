import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.utils.validation import check_is_fitted

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

   
 

  


    # Fetch data from GitHub
    url = "https://raw.githubusercontent.com/wisabd/BasicMLapp/master/buenos-aires-test-features.csv"
    response = requests.get(url)

    if response.status_code == 200:
        st.write("Data fetched successfully!")
        X_test = pd.read_csv(StringIO(response.text))
    else:
        st.error("Failed to fetch the file. Please check the URL or your internet connection.")
        st.stop()  # Stop execution if the file cannot be fetched

 
    # Extract X_train and y_train
    X_train = df["surface_covered_in_m2"].values.reshape(-1, 1)  # Reshape for sklearn
    y_train = df["price_aprox_usd"]

    # Baseline model (mean prediction)
    y_mean = y_train.mean()
    y_pred_baseline = [y_mean] * len(y_train)
    mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

    # Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    check_is_fitted(model)  # Ensure the model is fitted

    # Predictions on training data
    y_pred_training = model.predict(X_train)
    mae_training = mean_absolute_error(y_train, y_pred_training)

    # Predictions on test data
    features = ["surface_covered_in_m2"]
    X_test = X_test[features]
    y_pred_test = pd.Series(model.predict(X_test))

    # Display the regression equation
    intercept = round(model.intercept_, 2)
    coefficient = round(model.coef_[0], 2)
    st.write(f"### Linear Regression Equation")
    st.write(f"apt_price = {intercept} + {coefficient} * surface_covered")

    # Create Plotly figure
    fig2 = go.Figure()

    # Add scatter plot for actual data
    fig2.add_trace(
        go.Scatter(
            x=X_train.flatten(),  # Flatten X_train for Plotly
            y=y_train,
            mode='markers',
            name='Actual Data Points',
            marker=dict(color='blue')
    )

    # Add line plot for model predictions
    fig2.add_trace(
    go.Scatter(
        x=X_train.flatten(),  # Flatten X_train for Plotly
        y=model.predict(X_train),
        mode='lines',
        name='Post-iteration Prediction: Linear Model',
        line=dict(color='red')
    )

    # Update layout
    fig2.update_layout(
    title="Buenos Aires: Price vs. Area",
    xaxis_title="Area [Sq metres]",
    yaxis_title="Price [USD]",
    legend_title="Legend"
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig2)

    
  
    


else:
    st.write('In the notg;')
    print("Failed to fetch the CSV file.")
