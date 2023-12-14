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
###########################################################setting page configs################################################################
st.set_page_config(
    page_title="Crystal Ball",
    page_icon="🔮",
    layout="wide",
)

######################################################Nav bar#################################33
selected = option_menu(
    menu_title="Project Crystal Ball",
    options=["Dashboard", "Classification", "Regression", "Sample Applications"],
    icons=["boxes", "layout-wtf", "graph-up-arrow", "grid-fill"],
    menu_icon="stack",
    orientation="horizontal")

###############################################################################DASHBORAD SECTION###########################################################################

if selected == "Dashboard":
    body="""
    <h1 >Welcome to Crystal Ball</h1>
    <p>Empower Your ML Journey: An Interactive Web AutoML App for Effortless 
    Model Training, Rapidly Transforming Ideas into Custom Machine Learning Apps</p>
    <h3>What is Crystal Ball?</h3>
    <p>Crystal Ball is a powerful tool catering to both seasoned data scientists and beginners. 
    It harnesses advanced algorithms and computational processes to reveal intricate patterns within your data. 
    This capability empowers machine learning models to make accurate predictions about future data.</p>
    <h3>Significance of Crystal Ball</h3>
    <p>The crystal ball in our project symbolizes the profound predictive power of machine learning models. 
    These models act as virtual crystal balls, allowing users to foresee patterns and trends in their datasets. 
    The project's high-tech theme reflects the sophistication of the underlying algorithms,
    making it a cutting-edge solution for data analysis.</p>
    """
    la1, la2 = st.columns(2)
    with la1:
        logo = Image.open("images/logo.png")
        st.image(logo.resize((480,480)))
    with la2:
        st.markdown(body,unsafe_allow_html=True)

    aa1, aa2 = st.columns(2)
    with aa1:
        body="""
        <h1>Instructions to Use</h1>
        <p>Using Crystal Ball is a straightforward process that involves a few key steps:</p>
        <h3>Upload Your Dataset:</h3>
        <p>Begin by uploading your dataset using the "Upload" section. Click on the "Upload Your Dataset" button and select your dataset file. 
        Crystal Ball supports various file formats, including CSV.</p>
        <h3>Get Your EDA</h3>
        <p>Once your dataset is uploaded, explore its characteristics through the "Profiling" section.
        Crystal Ball performs Exploratory Data Analysis (EDA) to provide insights into the dataset's structure, statistics, and patterns.</p>
        <h3>Train Your Model</h3>
        <p>Move on to the "Modelling" section to initiate the model training process. 
        Choose the target column for your classification or regression task.
        Crystal Ball automates the setup, compares different models, and selects the best-performing one.</p>
        <h3>Download Your Model:</h3>
        <p>After the model is trained, download it for use in your projects. Utilize the "Download" section to obtain the trained model file.</p> 
        """
        st.markdown(body, unsafe_allow_html=True)
    with aa2:
        pass
    st.markdown("""<h1 style="text-align:center">Algorithms Used for Crystal Ball</h1>""",unsafe_allow_html=True)
    st.markdown("""<p>The provided code uses the PyCaret library for AutoML tasks in both classification and regression scenarios.
     PyCaret internally utilizes a variety of machine learning algorithms, such as Decision Trees, 
     Random Forest, Gradient Boosting, Support Vector Machines, K-Nearest Neighbors, and more. 
     The library automatically selects and tunes algorithms based on dataset characteristics and user preferences.</p><p>It's worth noting that users 
     can customize the selection of
     algorithms and other parameters according to their specific requirements and preferences.""",unsafe_allow_html=True)
    st.markdown("""<h1>Explore More</h1>""",unsafe_allow_html=True)
    bb1, bb2 = st.columns(2)
    with bb1:
        body = """
        <h3>Explore Sample Applications:</h3>
        <p>If you're new to machine learning or want to explore predefined use cases, check out the "Sample Applications" section. Crystal Ball provides sample 
        applications like "Player Price Prediction," "Weather Prediction," and more. 
        Simply follow the prompts and make predictions based on the provided models.</p>
        """
        st.markdown(body, unsafe_allow_html=True)
    with bb2:
        body="""
        <h3>Navigate Between Sections:</h3>
        <p>Use the navigation bar on the left to switch between different sections - "Home," "Dashboard," "Classification," "Regression," and "Sample Applications."
        Now you're ready to unleash the power of Crystal Ball for your machine learning tasks.
         Whether you're a data science expert or just starting, Crystal Ball makes the process intuitive and efficient.</p>
        """
        st.markdown(body,unsafe_allow_html=True)


###################################################################Classification Trainer#################################################################################

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

####################################################################Regression Trainer###########################################################################################

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

########################################################################SAmple Applications###############################################################################

if selected == "Sample Applications":
    options = option_menu(
    menu_title=None,
    options=["About","Player Price Prediction", "Weather prediction", "Predicto", "Anything"],
    icons=["body-text","dribbble", "cloud-sun-fill", "coin", "hourglass"],
    orientation="horizontal")
####################About sample apps
    if options == "About":
        la1, la2 = st.columns(2)
        with la1:
            st.image("images/robot.gif")
        with la2:
            h1 = """
            <h1 style="padding-top:65px">
            Sample Model Significance
            </h1>
            <p>Crystal Ball provides sample applications like "Player Price Prediction," "Weather Prediction," and more. 
            Simply follow the prompts and make predictions based on the provided models.
            </p>
            <h3>Models Provided</h3>
            <ul><li>Player Price Prediction</li><li>Bitcoin Prediction</li><li>Emplyee Churn Prediction</li><li>Anything else</li></ul>
            <p>The library automatically selects and tunes algorithms based on dataset characteristics and user preferences.</p>
            """
            st.markdown(h1,unsafe_allow_html=True)
        st.markdown("""<h1>Use Cases of Given Sample Models</h1>""",unsafe_allow_html=True)
        ab1, ab2 = st.columns(2)
        with ab1:
            body = """<h3>Bitcoin Price Prediction:</h3>
            <h5>Use Case:</h5>
            <p>Crystal Ball includes a sample model for predicting Bitcoin prices, illustrating the application of machine learning in financial markets.</p>
            <h5>How it Works:</h5>
            <p>The model is trained on historical Bitcoin price data along with relevant features such as trading volume,
            market sentiment, and other market indicators. The trained model can then forecast future Bitcoin prices based on the input data.</p>
            <h5>Significance:</h5>
            <p>Predicting cryptocurrency prices is a challenging task due to market volatility. 
            This sample application showcases how machine learning can analyze complex data to provide insights into potential price movements.</p>"""
            st.markdown(body,unsafe_allow_html=True)
        with ab2:
            st.image("images/bitcoin.jpeg")
        ac1,ac2 = st.columns(2)
        with ac1:
            logo = Image.open("images/football.jpg")
            st.image(logo.resize((500, 450)))
        with ac2:
            body = """
            <h3>Player Price Prediction:</h3>
            <h5>Use Case:</h5>
            <p> Crystal Ball offers a sample application for predicting player salaries, demonstrating the application of machine learning in sports analytics.</p>
            <h5>How it Works:</h5>
            <p>The model is trained on historical data related to players, including factors like player rating, team, position, draft year, and more. By learning
            from historical salary data, 
            the model can predict the salaries of players based on specific input features.</p>
            <h5>Significance:</h5>
            <p>This sample application highlights how machine learning can contribute to talent valuation in sports. 
            It can be valuable for sports teams, agents, and analysts in making informed decisions about player contracts and team composition.</p>
            """
            st.markdown(body,unsafe_allow_html=True)

################PLAYER PRICE PREDICTION###########################3
    if options == "Player Price Prediction":
        model = pickle.load(open('pretrained models/model.sav', 'rb'))
        st.title('Player Salary Prediction')
        st.sidebar.header('Player Data')
        image = Image.open('images/football.jpg')
        st.image(image, '')

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
####################agla vala #################################
        if options == "Weather prediction":
            pass

        if options == "pridicto":
            pass

        if options == "Anything":
            pass

