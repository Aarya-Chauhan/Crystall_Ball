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
    body = """
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
        st.image(logo.resize((600, 480)))
    with la2:
        st.markdown(body, unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid grey'>", unsafe_allow_html=True)
    aa1, aa2 = st.columns(2)
    with aa1:
        body = """
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
    st.markdown("""<h1 style="text-align:center">How Crystal Ball Works</h1>""", unsafe_allow_html=True)
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
     few lines of code.</p>""", unsafe_allow_html=True)
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
    st.markdown("""<h1>Features of Crystal Ball :</h1>""", unsafe_allow_html=True)
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
        st.markdown(body, unsafe_allow_html=True)
    with zz2:
        body = """
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
        st.markdown(body, unsafe_allow_html=True)
    st.markdown("<hr style='border:1px dashed black'>", unsafe_allow_html=True)
    st.markdown(
        """<br><b style=" margin-bottom:0px padding-bottom:0px">Crystall Ball aims to make machine learning accessible to all as an Intutive user friendly tool which lets everyone use powers of Machine Learning irrespective of their coding and technical skills.</b>""",
        unsafe_allow_html=True)

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
        options=["About", "Employee Churn Analysis", "Bitcoin Price Prediction", "Player Price Prediction",
                 "Announcements"],
        icons=["body-text", "briefcase", "coin", "dribbble", "megaphone"],
        orientation="horizontal")
    ####################Announcements
    if options == "Announcements":
        col1, col2, col3 = st.columns([2, 6, 2])

        with col1:
            st.write("")

        with col2:
            st.markdown("""<br>""", unsafe_allow_html=True)
            st.image("images/Core.gif")
            st.markdown("""<h1>More Updates On The Way . . . . """, unsafe_allow_html=True)

        with col3:
            st.write("")
    ########################About section
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
            st.markdown(h1, unsafe_allow_html=True)
        st.markdown("<hr style='border:1px solid grey'>", unsafe_allow_html=True)
        st.markdown("""<h1>Introduction to our sample apps.</h1>""", unsafe_allow_html=True)
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
            st.markdown(body, unsafe_allow_html=True)
        with ab1:
            ins = Image.open("images/bitcoin.jpg")
            st.image(ins.resize((650, 420)))
        st.markdown("<hr style='border:1px solid grey'>", unsafe_allow_html=True)
        ac1, ac2 = st.columns(2)
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
            st.markdown(body, unsafe_allow_html=True)
        st.markdown("<hr style='border:1px solid grey'>", unsafe_allow_html=True)

    ################Player Price  PREDICTION###########################3
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

        ################Employee Churn Analysis###########################3

    if options == "Employee Churn Analysis":
        import pandas as pd
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.cluster import KMeans
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn import metrics
        import plotly.express as px

        # Load the pre-trained model
        model = pickle.load(open('pretrained models/employee_churn_model.pkl', 'rb'))

        # Load the CSV file
        if os.path.exists('./datasets/HR_comma_sep.csv'):
            df = pd.read_csv('datasets/HR_comma_sep.csv', index_col=None)

        features_for_prediction = ['satisfaction_level', 'last_evaluation', 'number_project',
                                   'average_montly_hours', 'time_spend_company', 'Work_accident',
                                   'promotion_last_5years', 'Departments ', 'salary']


        # Function to handle user input for clustering analysis
        def user_report():
            # Assuming df is your DataFrame containing historical employee data
            if os.path.exists('./datasets/HR_comma_sep.csv'):
                df = pd.read_csv('datasets/HR_comma_sep.csv', index_col=None)

            # Create an empty dictionary to store user input data
            user_data = {}

            # Iterate over features to get user input
            for feature in features_for_prediction:
                if df[feature].dtype == 'float64':
                    # If the feature is of type float, use float values for slider
                    user_data[feature] = st.sidebar.slider(f'Select {feature}', float(df[feature].min()),
                                                           float(df[feature].max()), float(df[feature].mean()))
                elif df[feature].dtype == 'int64':
                    # If the feature is of type int, use int values for slider
                    user_data[feature] = st.sidebar.slider(f'Select {feature}', int(df[feature].min()),
                                                           int(df[feature].max()), int(df[feature].mean()))
                else:
                    # Handle other data types as needed
                    user_data[feature] = st.sidebar.text_input(f'Enter {feature}', df[feature].iloc[0])

            return pd.DataFrame(user_data, index=[0])


        def descriptive_statistics():
            st.title(" Employee Attrition Analysis")
            b = df.describe()
            st.dataframe(b)


        # Apply label encoding to categorical columns
        df['Departments '] = LabelEncoder().fit_transform(df['Departments '])
        df['salary'] = LabelEncoder().fit_transform(df['salary'])

        # Handle missing values if any
        if df.isnull().any().any():
            df = df.fillna(df.mean())

        # Streamlit App
        with st.sidebar:
            st.image('images/employee churn.jpg')
            st.title("Employee Churn")
            choice = st.radio("Navigation",
                              ["Profiling", "Stayed vs. Left: Employee Data Comparison",
                               "Descriptive Statistics Overview",
                               "Employees Left", "Show Value Counts", "Number of Projects Distribution",
                               "Time Spent in Company",
                               "Employee Count by Features", "Clustering of Employees who Left",
                               "Employee Clustering Analysis", "Predict Churn"])
            st.info(
                "Employee Churn App provides a user-friendly interface for HR professionals and data enthusiasts to "
                "explore and gain insights from employee data, with a focus on predicting and understanding employee "
                "turnover.")

        if choice == "Profiling":
            st.title("Data Profiling Dashboard")
            a = df.head()
            st.dataframe(a)

        if choice == "Stayed vs. Left: Employee Data Comparison":
            st.title("Employee Retention Analysis: Comparing Characteristics of Stayed and Left Groups")
            left = df.groupby('left')
            b = left.mean()
            st.dataframe(b)

        if choice == "Descriptive Statistics Overview":
            descriptive_statistics()

        if choice == "Employees Left":
            st.title("Data Visualization")
            left_count = df.groupby('left').count()
            st.bar_chart(left_count['satisfaction_level'])

        if choice == "Show Value Counts":
            st.title("Employee Left Counts")
            left_counts = df.left.value_counts()
            st.write(left_counts)
            st.bar_chart(left_counts)

        if choice == "Number of Projects Distribution":
            st.title("Employees' Project Distribution")
            num_projects = df.groupby('number_project').count()
            plt.bar(num_projects.index.values, num_projects['satisfaction_level'])
            plt.xlabel('Number of Projects')
            plt.ylabel('Number of Employees')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

        if choice == "Time Spent in Company":
            st.title("Data Visualization")
            time_spent = df.groupby('time_spend_company').count()
            plt.bar(time_spent.index.values, time_spent['satisfaction_level'])
            plt.xlabel('Number of Years Spent in Company')
            plt.ylabel('Number of Employees')
            st.pyplot()

        if choice == "Employee Count by Features":
            st.title("Data Visualization")
            features = ['number_project', 'time_spend_company', 'Work_accident', 'left', 'promotion_last_5years',
                        'Departments ', 'salary']

            fig, axes = plt.subplots(4, 2, figsize=(10, 15))

            for i, j in enumerate(features):
                row, col = divmod(i, 2)
                sns.countplot(x=j, data=df, ax=axes[row, col])
                axes[row, col].tick_params(axis="x", rotation=90)
                axes[row, col].set_title(f"No. of Employees - {j}")

            plt.tight_layout()
            st.pyplot(fig)

        if choice == "Clustering of Employees who Left":
            X = df[['satisfaction_level', 'last_evaluation', 'number_project',
                    'average_montly_hours', 'time_spend_company', 'Work_accident',
                    'promotion_last_5years', 'Departments ', 'salary']]
            y = df['left']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            gb = GradientBoostingClassifier()
            gb.fit(X_train, y_train)
            y_pred = gb.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)
            st.title("Gradient Boosting Classifier Model Evaluation")
            st.write("Accuracy:", accuracy)
            st.write("Precision:", precision)
            st.write("Recall:", recall)

            y_pred_all = gb.predict(X)
            diff_all_df = pd.DataFrame({
                'Sample': range(len(y)),
                'Actual': y,
                'Predicted': y_pred_all
            })
            diff_all_df['Correct'] = (diff_all_df['Actual'] == diff_all_df['Predicted']).astype(int)
            diff_counts = diff_all_df.groupby('Correct').size().reset_index(name='Count')
            fig_diff_all = px.bar(diff_counts, x='Correct', y='Count', color='Correct',
                                  labels={'Correct': 'Prediction Correctness', 'Count': 'Number of Samples'},
                                  title='Actual vs Predicted for All Data',
                                  color_discrete_map={0: 'red', 1: 'green'})
            fig_diff_all.update_layout(showlegend=False)
            st.plotly_chart(fig_diff_all)

        if choice == "Employee Clustering Analysis":
            st.title("Employee Clustering Analysis")
            user_data = user_report()
            user_data['Departments '] = LabelEncoder().fit_transform(user_data['Departments '])
            user_data['salary'] = LabelEncoder().fit_transform(user_data['salary'])
            if user_data.isnull().any().any():
                user_data = user_data.fillna(user_data.mean())
            features_for_clustering = ['satisfaction_level', 'last_evaluation', 'number_project',
                                       'average_montly_hours', 'time_spend_company', 'Work_accident',
                                       'promotion_last_5years', 'Departments ', 'salary']
            scaler = StandardScaler()
            user_data[['satisfaction_level', 'last_evaluation', 'average_montly_hours']] = scaler.fit_transform(
                user_data[['satisfaction_level', 'last_evaluation', 'average_montly_hours']])
            X = df[features_for_clustering]
            num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df['Cluster'] = kmeans.fit_predict(X)
            fig_clusters = px.scatter_3d(df, x='satisfaction_level', y='last_evaluation', z='average_montly_hours',
                                         color='Cluster', opacity=0.7, title='Employee Clusters')
            st.plotly_chart(fig_clusters)

        if choice == "Predict Churn":
            st.title("Employee Churn Prediction")
            user_report_data = {
                'satisfaction_level': st.sidebar.slider('Satisfaction Level', 0.0, 1.0, 0.5),
                'last_evaluation': st.sidebar.slider('Last Evaluation', 0.0, 1.0, 0.5),
                'number_project': st.sidebar.slider('Number of Projects', 2, 7, 4),
                'average_montly_hours': st.sidebar.slider('Average Monthly Hours', 80, 300, 160),
                'time_spend_company': st.sidebar.slider('Time Spent in Company', 2, 10, 3),
                'Work_accident': st.sidebar.selectbox('Work Accident', [0, 1]),
                'promotion_last_5years': st.sidebar.selectbox('Promotion in Last 5 Years', [0, 1]),
                'Departments ': st.sidebar.selectbox('Department', df['Departments '].unique()),
                'salary': st.sidebar.selectbox('Salary', df['salary'].unique())
            }
            user_data = pd.DataFrame(user_report_data, index=[0])
            st.header('Employee Data for Prediction')
            st.write(user_data)
            features_for_prediction = ['satisfaction_level', 'last_evaluation', 'number_project',
                                       'average_montly_hours', 'time_spend_company', 'Work_accident',
                                       'promotion_last_5years', 'Departments ', 'salary']
            missing_columns = set(features_for_prediction) - set(user_data.columns)
            if missing_columns:
                st.error(f"Columns {missing_columns} not found in user data.")
            else:
                X_pred = user_data[features_for_prediction]
                X_pred.columns = ['satisfaction_level', 'last_evaluation', 'number_project',
                                  'average_montly_hours', 'time_spend_company', 'Work_accident',
                                  'promotion_last_5years', 'Departments ', 'salary']
                churn_prediction = model.predict(X_pred)
                st.subheader('Churn Prediction Result')
                st.write(churn_prediction)

    ################Bitcoin Prediction###########################3

    if options == "Bitcoin Price Prediction":
        import os
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sb
        import streamlit as st

        if os.path.exists('./datasets/BTC-USD.csv'):
            df = pd.read_csv('datasets/BTC-USD.csv', index_col=None)

        with st.sidebar:
            st.image("images/bitcoin.jpg")
            st.title("Phase II Project Crystal Ball")
            choice = st.radio("Navigation",
                              ["Shape Dimension", "Bitcoin close price", "frequency graph", "boxplot graph",
                               "Pie Chart", "Variation in the price of cryptocurrency", "Distribution Plots"])
            st.info("This interface is built for analysing and comparing models main Dashboard of Project Crystal Ball")

        if choice == "Shape Dimension":
            st.title("Describes Shape Dimension")
            a = df.describe()
            st.dataframe(a)

        if choice == "Bitcoin close price":
            st.title("Bitcoin close price")
            fig, ax = plt.subplots(figsize=(25, 15))
            ax.plot(df['Close'])
            ax.set_title('Bitcoin Close Price Graph', fontsize=40)
            ax.set_ylabel('Price in INR')
            st.pyplot(fig)

        # Check user's choice and generate the appropriate plot

        if choice == "frequency graph":
            st.title("Frequency Graph")
            df = pd.DataFrame({
                'year': np.random.choice([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021], size=100),
                'Open': np.random.randn(100) * 10 + 170,
                'High': np.random.randn(100) * 10 + 180,
                'Low': np.random.randn(100) * 10 + 160,
                'Close': np.random.randn(100) * 10 + 175,
                'Adj Close': np.random.randn(100) * 10 + 175,
            })

            # Group data by 'year' and calculate the mean
            data_grouped = df.groupby('year').mean()

            # Display available columns for user selection
            selected_columns = st.multiselect("Select columns to plot", ['Open', 'High', 'Low', 'Close'])

            # Check if selected columns exist in the grouped DataFrame
            valid_columns = [col for col in selected_columns if col in data_grouped.columns]

            if not valid_columns:
                st.error("None of the selected columns exist in the grouped DataFrame.")
            else:
                # Create grouped bar plots
                st.bar_chart(data_grouped[valid_columns])

        if choice == "Pie Chart":
            df['open-close'] = df['Open'] - df['Adj Close']
            df['low-high'] = df['Low'] - df['High']

            # Create a 'target' column for binary classification labels
            df['target'] = np.where(df['Adj Close'].shift(-1) > df['Adj Close'], 1, 0)

            # Display a pie chart to visualize the distribution of 'target' values
            st.title("Target Distribution Pie Chart")
            fig, ax = plt.subplots()
            ax.pie(df['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)

        if choice == "boxplot graph":
            st.title("Boxplot Graph")
            features = ['Open', 'High', 'Low', 'Adj Close']
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            for i, col in enumerate(features):
                sb.boxplot(df[col], ax=axes[i // 2, i % 2])
                axes[i // 2, i % 2].set_title(col)
            st.pyplot(fig)

        if choice == "Variation in the price of cryptocurrency":
            st.title("Variation in the price of cryptocurrency")
            df = pd.DataFrame({
                'Open': np.random.randn(100) * 10 + 170,
                'Close': np.random.randn(100) * 10 + 180,
                'Adj Close': np.random.randn(100) * 10 + 175,
                'High': np.random.randn(100) * 10 + 185,
                'Low': np.random.randn(100) * 10 + 165
            })
            # User choice
            choice = st.selectbox("Select Plot Type", ["Variation in the price of cryptocurrency"])

            # Display available columns for user selection
            selected_column = st.selectbox("Select a column to plot", df.columns)

            # Create Seaborn plots
            fig, ax = plt.subplots(figsize=(10, 6))

            # Check if the selected column exists in the DataFrame and is numeric
            if selected_column in df.columns:
                if pd.api.types.is_numeric_dtype(df[selected_column]):
                    sns.distplot(df[selected_column], ax=ax)
                    st.pyplot(fig)
                else:
                    st.error(f"Column '{selected_column}' is not numeric in the DataFrame.")
            else:
                st.error(f"Column '{selected_column}' does not exist in the DataFrame.")
                st.write("Available columns:", df.columns)

        if choice == "Distribution Plots":
            st.title("Distribution Plots")

            # Display distribution plots for selected features
            features = ['Open', 'High', 'Low', 'Close']

            # Create subplots
            fig, axs = plt.subplots(2, 2, figsize=(20, 10))

            for i, col in enumerate(features):
                row, col_num = divmod(i, 2)

                # Check if the column exists in the DataFrame
                if col in df.columns:
                    sb.histplot(df[col], kde=True, ax=axs[row, col_num])
                    axs[row, col_num].set_title(f'Distribution of {col}')
                else:
                    st.warning(f"Column '{col}' not found in the DataFrame.")

            # Display the plots in Streamlit
            st.pyplot(fig)

            df = pd.DataFrame({
                'year': np.random.choice([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021], size=100),
                'Open': np.random.randn(100) * 10 + 170,
                'High': np.random.randn(100) * 10 + 180,
                'Low': np.random.randn(100) * 10 + 160,
                'Close': np.random.randn(100) * 10 + 175,
                'Adj Close': np.random.randn(100) * 10 + 175,
            })

            # Group data by 'year' and calculate the mean
            data_grouped = df.groupby('year').mean()

            # Streamlit app
            st.title("Grouped Bar Plots")

            # Display available columns for user selection
            selected_columns = st.multiselect("Select columns to plot", ['Open', 'High', 'Low', 'Close'])

            # Check if selected columns exist in the grouped DataFrame
            valid_columns = [col for col in selected_columns if col in data_grouped.columns]
            st.bar_chart(data_grouped[valid_columns])
