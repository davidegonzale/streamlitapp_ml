import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
import matplotlib.pyplot as plt


siteHeader = st.beta_container()
dataExploration = st.beta_container()
newFeatures = st.beta_container()
modelTraining = st.beta_container()


# configuration
st.set_option('deprecation.showfileUploaderEncoding', False)

# Add a sidebar
st.sidebar.subheader("Visualization Settings")

# Setup file upload
uploaded_file = st.sidebar.file_uploader(
                        label="Upload your CSV or Excel file. (200MB max)",
                         type=['csv', 'xlsx'])


# SITE HEADER AND INTRO TO THE PROJECT
with siteHeader:
    st.title('Machine Learning Models Comparison')
    st.markdown('''The dataset used for this project was taken out of the following URL:
**https://www.kaggle.com/sakshigoyal7/credit-card-customers**''')

# DF EXPLORATION AND VISUALIZATION PLUS ATTEMPT AT HEAT-MAP
with dataExploration:
    st.header('Dataset explore')
    st.text('Here you can visualize the uploaded Data Set')

    global df
    if uploaded_file is not None:
        print(uploaded_file)
        print("hello")

        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            df = pd.read_excel(uploaded_file)

    global numeric_columns
    global non_numeric_columns
    try:
        st.write(df)
        numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
        non_numeric_columns = list(df.select_dtypes(['object']).columns)
        non_numeric_columns.append(None)
        print(non_numeric_columns)
    except Exception as e:
        print(e)
        st.markdown("**Please upload file to the application.**")

    # add a select widget to the side bar
    chart_select = st.sidebar.selectbox(
        label="Select the chart type",
        options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot']
    )

    if chart_select == 'Scatterplots':
        st.sidebar.subheader("Scatterplot Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
            plot = px.scatter(data_frame=df, x=x_values, y=y_values, color=color_value)
            # display the chart
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Lineplots':
        st.sidebar.subheader("Line Plot Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
            plot = px.line(data_frame=df, x=x_values, y=y_values, color=color_value)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Histogram':
        st.sidebar.subheader("Histogram Settings")
        try:
            x = st.sidebar.selectbox('Feature', options=numeric_columns)
            bin_size = st.sidebar.slider("Number of Bins", min_value=10,
                                         max_value=100, value=40)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
            plot = px.histogram(x=x, data_frame=df, color=color_value)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Boxplot':
        st.sidebar.subheader("Boxplot Settings")
        try:
            y = st.sidebar.selectbox("Y axis", options=numeric_columns)
            x = st.sidebar.selectbox("X axis", options=non_numeric_columns)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
            plot = px.box(data_frame=df, y=y, x=x, color=color_value)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

# Intecorrelation map

    if st.button('Intercorrelation Heatmap'):
        st.header('Intercorrelation Matrix Heatmap')
        df = pd.read_csv(uploaded_file)

        corr = pd.concat([cat_df, y], axis=1).corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, vmin=-1.0, cmap='crest', linewidths=.5)
        plt.show()

with newFeatures:
    st.header('New features I came up with')
    st.text('Nothing here so far')

with modelTraining:
    st.header('Model training')
    st.text('In this section you can select the Hyperparameters')
