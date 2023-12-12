import streamlit as st
import pickle
import numpy as np
from streamlit_card import card
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
from streamlit_image_select import image_select

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
        logo = Image.open("images/logo.png")
        st.image(logo.resize((480,480)))
    with la2:
        st.title("Crystal Ball")
        st.write(des)

# Classification Trainer Code

if selected == "Classification":

    if os.path.exists('./dataset.csv'):
        df = pd.read_csv('dataset.csv', index_col=None)

    with st.sidebar:
        st.image('images/Core.gif')
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
        st.image("images/Core.gif", use_column_width="always")
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
    options = option_menu(
    menu_title=None,
    options=["Player Price Prediction", "Weather prediction", "Predicto", "Anything"],
    icons=["dribbble", "cloud-sun-fill", "coin", "hourglass"],
    orientation="horizontal")

    if options == "Player Price Prediction":
        model = pickle.load(open('pretrained models/model.sav', 'rb'))
        st.title('Player Salary Prediction')
        st.sidebar.header('Player Data')
        image = Image.open('images/football.jpg')
        st.image(image, '')


        # FUNCTION
        def user_report():
            rating = st.sidebar.slider('Rating', 50, 100, 1)
            jersey = st.sidebar.slider('Jersey', 0, 100, 1)
            team = st.sidebar.slider('Team', 0, 30, 1)
            position = st.sidebar.slider('Position', 0, 10, 1)
            country = st.sidebar.slider('Country', 0, 3, 1)
            draft_year = st.sidebar.slider('Draft Year', 2000, 2020, 2000)
            draft_round = st.sidebar.slider('Draft Round', 1, 10, 1)
            draft_peak = st.sidebar.slider('Draft Peak', 1, 30, 1)

            user_report_data = {
                'rating': rating,
                'jersey': jersey,
                'team': team,
                'position': position,
                'country': country,
                'draft_year': draft_year,
                'draft_round': draft_round,
                'draft_peak': draft_peak
            }
            report_data = pd.DataFrame(user_report_data, index=[0])
            return report_data


        user_data = user_report()
        st.header('Player Data')
        st.write(user_data)

        salary = model.predict(user_data)
        st.subheader('Player Salary')
        st.subheader('$' + str(np.round(salary[0], 2)))

        if options == "Weather prediction":
            pass

        if options == "pridicto":
            pass

        if options == "Anything":
            pass

