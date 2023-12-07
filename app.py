import streamlit as st
from operator import index
import plotly.express as px
from pycaret.classification import setup, compare_models, pull, save_model, load_model
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import ydata_profiling
from PIL import Image
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Crystal Ball",
    page_icon="🔮",
    layout="wide",

)

des = """The "Crystal Ball" project is a web-based AutoML application. This application is designed to train Machine 
Learning models on a provided dataset. The images you see represent various aspects of the project. The crystal ball 
image symbolizes the predictive power of machine learning models, which can "see" patterns in data and make 
predictions about future data. The high-tech theme of the images reflects the advanced algorithms and computational 
processes involved in training these models. The web-based nature of the application means that it can be accessed 
from anywhere with an internet connection, making it highly accessible and user-friendly. Users can upload their 
datasets, select the type of model they want to train, and then let the application handle the rest. The application 
will automatically preprocess the data, select the best model parameters, and train the model. Once the model is 
trained, users can download it for use in their own projects or use it directly within the application to make 
predictions on new data. Overall, the "Crystal Ball" project represents a powerful tool for anyone looking to 
leverage the power of machine learning, whether they are experienced data scientists or beginners just starting out 
in the field."""

# Nav Bar
selected = option_menu(
    menu_title="Project Crystal Ball",
    options=["Dashboard", "Classification", "Regression", "Sample Applications"],
    icons=["boxes", "layout-wtf", "graph-up-arrow", "grid-fill"],
    menu_icon="stack",
    orientation="horizontal")

if selected == "Dashboard":
    la1, la2 = st.columns(2)
    with la1:
        logo = Image.open("logo.png")
        st.image(logo.resize((480,480)))
    with la2:
        st.title("Crystal Ball")
        st.write(des)

# Classification Trainer Code

if selected == "Classification":

    if os.path.exists('./dataset.csv'):
        df = pd.read_csv('dataset.csv', index_col=None)

    with st.sidebar:
        st.image("Core.gif")
        st.title("Crystal Ball : Classification Trainer")
        choice = st.radio(
            "Workflow 👇", ["Upload", "Profiling", "Modelling", "Download"])
        st.info("This is an AutoML app for Classification problems just upload a dataset and go through the selection "
                "steps only this time let our app do all the hardworking.")

    if choice == "Upload":
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset")
        if file:
            df = pd.read_csv(file, index_col=None)
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)

    if choice == "Profiling":
        st.title("Exploratory Data Analysis")
        profile_df = df.profile_report()
        st_profile_report(profile_df)

    if choice == "Modelling":
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):
            setup(df, target=chosen_target)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')

    if choice == "Download":
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")

# Regression Trainer Code

if selected == "Regression":

    if os.path.exists('./dataset.csv'):
        df = pd.read_csv('dataset.csv', index_col=None)

    with st.sidebar:
        st.image("Core.gif", use_column_width="always")
        st.title("Crystal Ball : Regression Trainer")
        choice = st.radio(
            "Workflow 👇", ["Upload", "Profiling", "Modelling", "Download"])
        st.info("This is an AutoML app for Regression problems just upload a dataset and go"
                " through the selection steps only this time let our app do all the hardworking.")

    if choice == "Upload":
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset")
        if file:
            df = pd.read_csv(file, index_col=None)
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)

    if choice == "Profiling":
        st.title("Exploratory Data Analysis")
        profile_df = df.profile_report()
        st_profile_report(profile_df)

    if choice == "Modelling":
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        if st.button('Run Modelling'):
            setup(df, target=chosen_target)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')

    if choice == "Download":
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")

if selected == "Sample Applications":
    pass
