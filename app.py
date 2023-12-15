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
        logo = Image.open("images/logo.jpg")
        st.image(logo.resize((600,480)))
    with la2:
        st.markdown(body,unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid grey'>", unsafe_allow_html=True)
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
        ins = Image.open("images/instructions.jpg")
        st.image(ins.resize((700, 700)))
    st.markdown("<hr style='border:1px solid grey'>", unsafe_allow_html=True)
    st.markdown("""<h1 style="text-align:center">How Crystal Ball Works</h1>""",unsafe_allow_html=True)
    st.markdown("""<p>The provided code uses the PyCaret library for AutoML tasks in both classification and regression scenarios.
     PyCaret internally utilizes a variety of machine learning algorithms, such as Decision Trees, 
     Random Forest, Gradient Boosting, Support Vector Machines, K-Nearest Neighbors, and more. 
     The library automatically selects and tunes algorithms based on dataset characteristics and user preferences.</p>
     <p>It's worth noting that users 
     can customize the selection of
     algorithms and other parameters according to their specific requirements and preferences.</p>
     <p>The AutoML pipeline in the project is implemented by PyCaret. 
     It is an open-source, low-code machine-learning library in Python that automates machine-learning workflows.
      It is designed to make performing standard tasks in a machine learning project easy1. 
      PyCaret is a Python version of the Caret machine learning package in R. 
     It is popular because it allows models to be evaluated, compared, and tuned on a given dataset with just a 
     few lines of code.</p>""",unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid grey'>", unsafe_allow_html=True)
    st.markdown("""<h1>Explore More</h1>""", unsafe_allow_html=True)
    bb1, bb2 = st.columns(2)
    with bb1:
        body = """
        <h3>Explore Sample Applications:</h3>
        <p>If you're new to machine learning or want to explore predefined use cases, check out the "Sample Applications" section. Crystal Ball provides sample 
        applications like <ul><li> Player Price Prediction</li> <li>Bitcoin Price Prediction</li> and more. 
        Simply follow the prompts and make predictions based on the provided models.</p>
        """
        st.markdown(body, unsafe_allow_html=True)
        body2 = """
                <h3>Navigate Between Sections:</h3>
                <p>Use the navigation bar on the left to switch between different sections <ul> <li>Dashboard</li> <li>Classification</li> <li>Regression</li>  <li>Sample Applications</li></ul>
                Now you're ready to unleash the power of Crystal Ball for your machine learning tasks.
                 Whether you're a data science expert or just starting, Crystal Ball makes the process intuitive and efficient.</p>
                """
        st.markdown(body2, unsafe_allow_html=True)
    with bb2:
        logo = Image.open("images/mixoo.jpg")
        st.image(logo.resize((650, 500)))

    st.markdown("<hr style='border:1px solid grey'>", unsafe_allow_html=True)
    st.markdown("""<h1>Features of Crystal Ball :</h1>""",unsafe_allow_html=True)
    zz1, zz2 = st.columns(2)
    with zz1:
        body = """ 
        <ul style="list-style-type:none">
        <li><h3>Interactive Visualizations:</h3>Explore data patterns and model insights through interactive visualizations using advanced libraries like Plotly and seaborn. 
        Users can now interactively analyze predictions and gain a deeper understanding of their datasets with visuals like bargraphs, heatmaps and correlation matrices.</li>
        <br>
        <li><h3>Web Based ML Dashboard:</h3>Crystal ball is deployed as a web-app which can be accessed on any device,
         allowing users to train machine learning models and generate reporst on datsets on the go. This will greately improve its accessibility
          as well as it will make ML avaialble to all.</li>
         <br>
         <li><h3>Model Deployment:</h3>Final output of crystal ball is a trained models,so that you can deploy these models into any machine learning application easily. 
          These pretrained pickled models a can be easily integrated into language independent machine learning apps to be deployed in projects of all scales.</li>
        </ul>
        """
        st.markdown(body,unsafe_allow_html=True)
    with zz2:
        body="""
        <ul style="list-style-type:none">
        <li><h3>Automated EDA:</h3>The framework performs exploratory data analysis on all the input datasets to 
        generate a comprehensive report on data parameters, outlyers, null values and high-correlations between the 
        data items so that user has 
        understanding of dataset and based on it can decide to whether move forward to train models or not.</li>
        <br>
         <li><h3>Documentation and Tutorials:</h3>Navigate Crystal Ball with ease. The comprehensive documentation and tutorials guide users through the application's features, machine learning concepts, 
        and provide practical examples for enhanced user understanding.</li>
        <br>
         <li><h3>Model Statistics:</h3>The dashboard provides all performance stats for all trained models to correctly
          back the claims for best models which it provides in download section enabling the users to understand, 
          how a particular model is better performing than rest. </li>
        """
        st.markdown(body,unsafe_allow_html=True)
    st.markdown("<hr style='border:1px dashed black'>", unsafe_allow_html=True)
    st.markdown("""<br><b style=" margin-bottom:0px padding-bottom:0px">Crystall Ball aims to make machine learning accessible to all as an Intutive user friendly tool which lets everyone use powers of Machine Learning irrespective of their coding and technical skills.</b>""",unsafe_allow_html=True)


###################################################################Classification Trainer#################################################################################

if selected == "Classification":

    if os.path.exists('./dataset.csv'):
        df = pd.read_csv('dataset.csv', index_col=None)

    with st.sidebar:
        st.image('images/clss.jpg')
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
        st.image("images/reg.jpg", use_column_width="always")
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
    options=["About","Employee Churn Prediction", "Bitcoin Price Prediction", "Player Price Prediction", "Announcements"],
    icons=["body-text", "briefcase","coin","dribbble", "megaphone"],
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
            <p>Crystal ball also provides some sample models just to showcase how machine learning can really enable 
            you to see into the future by making 
            predictions on data by learning from it. These models are just a showcase for displaying the capabilities of 
            the models trained on the Crystal Ball dashboard.
            </p>
            <h3>Models Provided</h3>
            <ul><li>Employee Churn Prediction</li><li>Bitcoin Price Prediction</li><li>Player Price Prediction</li></ul>
            <p>The library automatically selects and tunes algorithms based on dataset characteristics and user preferences.</p>
            """
            st.markdown(h1,unsafe_allow_html=True)
        st.markdown("<hr style='border:1px solid grey'>", unsafe_allow_html=True)
        st.markdown("""<h1>Introduction to our sample apps.</h1>""",unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            ins = Image.open("images/employee churn.jpg")
            st.image(ins.resize((650, 500)))
        with cc2:
            body = """
            <h3>Employee Churn Analysis</h3>
            <h5>Use Case:</h5>
            <p>In the realm of workforce management, Crystal Ball presents a powerful application for Employee Churn Analysis,
             showcasing the practical application of machine learning in HR analytics.</p>
            <h5>How it Works:</h5>
            <p>Our model is meticulously trained on historical employee data, encompassing various factors such as job satisfaction,
             last evaluation, number of projects, average monthly hours, time spent in the company, work accidents, promotions in the 
             last 5 years, department, and salary.
             Leveraging this rich historical dataset, the model can predict employee churn based on these specific input features.</p>
            <h5>Significance:</h5>
            <p>This application serves as a beacon for the strategic use of machine 
            learning in understanding and predicting employee turnover. HR professionals, executives, 
            and analysts can harness the insights provided by the model to make informed decisions about 
            talent retention strategies, team composition, and overall workforce management. By identifying potential churn patterns, 
            organizations can proactively address employee satisfaction and engagement,
            ultimately contributing to a more stable and productive work environment.</p>"""

            st.markdown(body, unsafe_allow_html=True)
        st.markdown("<hr style='border:1px solid grey'>", unsafe_allow_html=True)
        ab1, ab2 = st.columns(2)
        with ab2:
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
        with ab1:
            ins = Image.open("images/bitcoin.jpg")
            st.image(ins.resize((650, 420)))
        st.markdown("<hr style='border:1px solid grey'>", unsafe_allow_html=True)
        ac1,ac2 = st.columns(2)
        with ac1:
            ins = Image.open("images/football (2).jpg")
            st.image(ins.resize((650, 420)))
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
        st.markdown("<hr style='border:1px solid grey'>", unsafe_allow_html=True)



################PLAYER PRICE PREDICTION###########################3
    if options == "Player Price Prediction":
        model = pickle.load(open('pretrained models/model.sav', 'rb'))
        st.title('Player Salary Prediction')
        st.sidebar.header('Player Data')
        image = Image.open('images/football (2).jpg')
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
    if options == "Announcements":
        col1, col2, col3 = st.columns([2, 6, 2])

        with col1:
            st.write("")

        with col2:
            st.markdown("""<br>""",unsafe_allow_html=True)
            st.image("images/Core.gif")
            st.markdown("""<h1>More Updates On The Way . . . . """,unsafe_allow_html=True)

        with col3:
            st.write("")







