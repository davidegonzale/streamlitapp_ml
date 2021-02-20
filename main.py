# IMPORT LIBRARIES
import streamlit as st
import pandas as pd
import numpy as np

import plotly_express as px
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# IGNORE WARNINGS
import warnings
warnings.filterwarnings("ignore")

st.set_option('deprecation.showPyplotGlobalUse', False)


# SEPARATE THE PAGE INTO DIFFERENT CONTAINERS; THIS IS FOR STREAMLIT AND BETTER VISUALIZATION
siteHeader = st.beta_container()
dataExploration = st.beta_container()
newFeatures = st.beta_container()
modelTraining = st.beta_container()
theEnd = st.beta_container()


# CONFIGURATION
st.set_option('deprecation.showfileUploaderEncoding', False)

# SIDEBAR
st.sidebar.subheader("Visualization Settings")

# FILE UPLOAD
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
    st.header('Introduction to the Data set')
    st.markdown('''Below you can have a first glimpse of what the Data set looks like,
    and play around with charts to visualize your data.''')

    global df
    if uploaded_file is not None:
        print(uploaded_file)
        print("hello")

        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            df = pd.read_excel(uploaded_file)

    # THE FOLLOWING SET OF FUNCTIONS ARE FOR LATER WHEN MODELLING
    # WE ENCODE FIRST TO ASSIGN "VALUES" TO OUR CATEGORICAL DATA COLUMNS

    def binary_encoding(df, column, positive_value):
        df = df.copy()
        df[column] = df[column].apply(lambda x: 1 if x == positive_value else 0)
        return df

    def ordinal_encoding(df, column, ordering):
        df = df.copy()
        df[column] = df[column].apply(lambda x: ordering.index(x))
        return df

    def onehot_encoding(df, column, prefix):
        df = df.copy()
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
        return df

    @st.cache
    def processed_data(df):
        df = df.copy()

        df = df.drop(df.columns[-2:], axis=1)

        df = df.drop('CLIENTNUM', axis=1)

        df = df.replace('Unknown', np.NaN)

        df['Education_Level'] = df['Education_Level'].fillna('Graduate')
        df['Income_Category'] = df['Income_Category'].fillna('Less than $40K')

    # BINARY COLUMNS ENCODING
        df = binary_encoding(df, 'Attrition_Flag', positive_value='Attrited Customer')
        df = binary_encoding(df, 'Gender', positive_value='M')

    # ENCODE ORDINAL COLUMNS
        education_ordering = [
            'Uneducated',
            'High School',
            'College',
            'Graduate',
            'Post-Graduate',
            'Doctorate'
        ]
        income_ordering = [
            'Less than $40K',
            '$40K - $60K',
            '$60K - $80K',
            '$80K - $120K',
            '$120K +'
        ]
        df = ordinal_encoding(df, 'Education_Level', ordering=education_ordering)
        df = ordinal_encoding(df, 'Income_Category', ordering=income_ordering)

    # ENCODE NOMINAL COLUMNS
        df = onehot_encoding(df, 'Marital_Status', prefix='St')
        df = onehot_encoding(df, 'Card_Category', prefix='Card')

    # SPLIT DF X & Y
        y = df['Attrition_Flag'].copy()
        X = df.drop('Attrition_Flag', axis=1).copy()

    # STANDARD SCALER X, this is going to give each column in X a mean 0 and a varience 1
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        return X, y

    X, y = processed_data(df)



################################################################################
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

    # SELECT WIDGET TO SIDEBAR
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


    st.markdown('''In the script it has been determined to return **X & y** as a Data frame. **Y** is the **Attrition flag**,
    which is what we would like to predict, and **X** is everthing else encoded, except the **Attrition flag**.''')

    if st.checkbox("Show X"):
        st.dataframe(X.head(10))

with newFeatures:
    st.header('Exploration of the data')
    st.text('Nothing here so far')

    # LETS CONCAT X FOR THE FOLLOWING VISUALIZATIONS
    cat_df = pd.concat([X.loc[:, ['Customer_Age', 'Months_on_book']], X.loc[:,'Credit_Limit':'Avg_Utilization_Ratio']], axis=1).copy()

    further_select = st.selectbox(
    label="Select the chart type",
    options=['Distplot', 'Boxplot', 'Pairplot', 'Heatmap'])

    if further_select == 'Distplot':
        plt.figure(figsize=(24, 15))

        for i in range(len(cat_df.columns) - 1):
            plt.subplot(3, 3, i + 1)
            sns.distplot(cat_df[cat_df.columns[i]])
        st.pyplot()

    if further_select == 'Boxplot':
        plt.figure(figsize=(24, 15))

        for i in range (len(cat_df.columns)-1):
            plt.subplot(3,3,i+1)
            sns.boxplot(cat_df[cat_df.columns[i]])
        st.pyplot()

    if further_select == 'Pairplot':
        plt.figure(figsize=(20, 20))
        sns.pairplot(cat_df)
        st.pyplot()

    if further_select == 'Heatmap':
        corr = pd.concat([cat_df, y], axis=1).corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, vmin=-1.0, cmap='YlGnBu', linewidths=.5)
        st.pyplot()

with modelTraining:
    st.header('Model training & Comparison')
    st.markdown('''For the purpose of this project, we will fit our data into 5 different
    models for comparison, these are: **Logistic Regression**, **Support Vector Machine**,
    **Decision Tree**, **MLP Classifier**, and **Random Forest Classifier**.''')

    # THIS DIVIDES THE SCREEN SECTION IN TWO COLUMNS
    selection_col, display_col = st.beta_columns(2)

    # GETTING TRAIN AND TEST READY
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)

    models = [
    LogisticRegression(),
    SVC(),
    DecisionTreeClassifier(),
    MLPClassifier(),
    RandomForestClassifier()
    ]

    for model in models:
        model.fit(X_train, y_train)

    model_names = [
    "   Logistic Regression",
    "Support Vector Machine",
    "         Decision Tree",
    "        MLP Classifier",
    "         Random Forest"
    ]



    selection_col.subheader("Here's the list of features: ")
    selection_col.write(X.columns)


    # RIGHT COLUMN OR RIGHT SIDE OF THE SCREEN SECTION
with display_col:
    st.subheader('Model Comparison: ')
    if st.checkbox("Show Model Scores"):
        for model, name in zip(models, model_names):
            st.write(name + ": {:.4f}%".format(model.score(X_test, y_test) * 100))

with theEnd:
    if st.button("Thank you!"):
        st.balloons()
