# -*- coding: utf-8 -*-
"""
Author: jhoernsch
This script uses the Streamlit library to make an interactive dashboard
I created it as part of a data science course with General Assembly
Thanks to Jonathan Bechtel and Andrew Riddle, my instructors, for being great teachers
"""

# import the libraries we need
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle


# let's make a function that loads the raw dataset
@st.cache     # this @ attribute tag tells streamlit not to re-run if this hasn't changed (a cache).  This must be directly above the function definition
def load_data():
    # read in the csv file into a dataframe
    df = pd.read_csv('https://raw.githubusercontent.com/jhoernsch/baby-b-nap-predictor/5d534803aea8fc25661b68b7ba95f7a019989b04/Baby_B_Activity_Anonymized.csv')
    return df


# let's make a function to group the dataframe by the specified x_val, then return the mean of the specified y_value
def group_data_mean(x_val, y_val):
    grouping = df.groupby(x_val)[y_val].mean()
    return grouping

# let's make a function to group the dataframe by the specified x_val, then return the mean of the specified y_value
def group_data_count(x_val, y_val):
    grouping = df.groupby(x_val)[y_val].count()
    return grouping


# let's make a function to load up a pickled model pipeline, which we created in a Jupyter notebook outside of this file
# @st.cache     # note that this threw an error, so we're commenting out for now
def load_model():
    # replace 'baby_b_nap_model.pkl' with the filepath if it is not saved in the same directory as this .py script
    pipe = pickle.load(open('baby_b_nap_model.pkl', 'rb'))
    return pipe


# let's make a function to load up a pickled naps dataframe, which we created in a Jupyter notebook outside of this file
# @st.cache     # note that this threw an error, so we're commenting out for now
def load_naps_df():
    # replace 'naps_dataframe.pkl' with the filepath if it is not saved in the same directory as this .py script
    naps = pickle.load(open('naps_dataframe.pkl', 'rb'))
    return naps


# create an overall title and some explanatory text for the site
st.title("Predicting Baby B's Nap Duration")
st.markdown("Using machine learning to predict how long Baby B will sleep.")



# let's populate a sidebar to house the user input controls

# make a radio button in the sidebar for app sections
# syntax: we pass the label for this array of radio buttons, and then pass a list of radio button labels
# note that the first item in the list will be the default, but we can change that with the `index` argument
app_section = st.sidebar.radio('What do you want to do?', ['Learn about the model', 
                                                           'Make a prediction', 
                                                           'Explore the data behind the model',
                                                           'Check out the code on GitHub'
                                                           ])


# create a df from the csv with the number of rows specified in the sidebar
df = load_data()


# depending on which section the user chose, display something different
if app_section == 'Learn about the model':
    st.header("The Background Story")
    st.markdown("Some people close to me recently had a baby.  For the sake of anonymity, we'll call these people Mom and Dad, and we'll call their new daughter **Baby B**.")
    st.markdown("Mom and Dad are wonderful people who aren't afraid of being nerds, so they used an app (called Baby Connect) to track Baby B's activity for the first five months of her life.")
    st.markdown("The Baby B activity dataset includes things like sleep, feeding, diaper changes, height, and weight, which Mom and Dad entered via the app.  Data entry wasn't always perfect during these early, sleep-deprived months.  This provided a real-world data-cleaning challenge.")
    st.markdown("Mom and Dad graciously offered this Baby B activity dataset, in return for some insights into what makes Baby B sleep longer.")
    st.markdown("We used a decision-tree-based Gradient Boosting Regressor model (called xgboost) to predict the duration of Baby B's naps.")
    st.header("The 4 Most Predictive Factors")
    st.markdown("The most important factors in predicting Baby B's nap durations are: \
                \n - The hour of the day (e.g., 4 pm)\
                \n - How long ago Baby B nursed (e.g., Baby B starts her nap 20 minutes after nursing)\
                \n - How long ago Baby B's last nap was (e.g., Baby B has been up for 3 hours)\
                \n - How long Baby B's last nap was (e.g., Baby B slept for 90 minutes in her last nap)"
                )
    st.markdown("By far, the most important factor in making an accurate prediction is the hour of the day (9 am, 7 pm, etc.).  \
                This single factor stood tall above the rest in terms of predictive ability, likely because the it tells us whether it's dark out,\
                as well as where Baby B is in her daily routine."
                )
    st.markdown("**If Mom and Dad want Baby B to sleep longer, they should try these things:** \
                \n - Put her to bed between 7 pm and midnight.\
                \n - Put her to bed shortly after nursing.\
                \n - Keep her up between naps for at least 45 minutes, but preferably for at least 3 hours.\
                \n - If Baby B's last nap was short, don't panic!  She'll sleep a bit longer than usual next time."
                )

    
    st.header("Try Making a Prediction!")
    st.markdown("Using the sidebar on the left, you can make a prediction of how long Baby B's nap will be, based on the most important factors that the model identified.")
    st.markdown("Note that sometimes, changing an input value may not affect the prediction.  This means the change is not important to the model.")


elif app_section == 'Make a prediction':
    # import the model that we created elsewhere in a Jupyter notebook
    model = load_model()
    
    # import the final naps dataframe that we created elsewhere in a Jupyter notebook
    naps = load_naps_df()
    
    # create a sidebar elements to capture user inputs for prediction factors
    user_hour = st.sidebar.number_input("What hour of the day is it? (Note: use 13 for 1 pm, 14 for 2 pm, etc.)", 
                                        min_value = 0, 
                                        max_value = 23, 
                                        value = 15,   # this is the value where the prediction changes if you bump in either direction, for wow factor
                                        step = 1)
    user_min_since_nurse = st.sidebar.number_input("How minutes has it been since Baby B nursed?", 
                                                   min_value = int(naps['Minutes since Previous Nursing'].min()),   # arguments must be an integer
                                                   max_value = int(naps['Minutes since Previous Nursing'].max()),   # arguments must be an integer 
                                                   value = 60,    # arguments must be an integer
                                                   step = 10)
    user_min_since_nap = st.sidebar.number_input("How minutes has it been since Baby B's last nap?", 
                                                 min_value = int(naps['Minutes since Previous Nap'].min()),   # arguments must be an integer
                                                 max_value = int(naps['Minutes since Previous Nap'].max()),   # arguments must be an integer 
                                                 value = 180,    # arguments must be an integer 
                                                 step = 15)
    user_prev_nap_duration = st.sidebar.number_input("How long was Baby B's last nap?", 
                                                     min_value = int(naps['Previous Nap Duration'].min()),   # arguments must be an integer, 
                                                     max_value = int(naps['Previous Nap Duration'].max()),   # arguments must be an integer, 
                                                     value = 90, 
                                                     step = 15)
    
    
    # create a tiny dataframe with the user-selected inputs
    # fill in the rest of the values with "typical" inputs, using either mode or mean
    user_specified_inputs = pd.DataFrame({
        'Caregiver': [naps['Caregiver'].mode()[0]],  # use the mode to get a typical value
        'Year': [naps['Year'].mode()[0]], # use the mode to get a typical value
        'Quarter': [naps['Quarter'].mode()[0]], # use the mode to get a typical value
        'Month': [naps['Month'].mode()[0]], # use the mode to get a typical value
        'Weekday': [naps['Weekday'].mode()[0]], # use the mode to get a typical value
        'Day': [naps['Day'].mode()[0]], # use the mode to get a typical value
        'Hour': [user_hour],
        'Is_Weekend': [naps['Is_Weekend'].mode()[0]], # use the mode to get a typical value
        'Previous Nap Duration': [user_prev_nap_duration],
        'Minutes since Previous Nap': [user_min_since_nap],
        'nap_duration_roll_count_24_hr': [naps['nap_duration_roll_count_24_hr'].mean()], # use the mean to get a typical value
        'Nursing Duration': [naps['Nursing Duration'].mean()], # use the mean to get a typical value
        'Minutes since Previous Nursing': [user_min_since_nurse],
        'Count of Feedings since Previous Nap': [naps['Count of Feedings since Previous Nap'].mode()[0]], # use the mode to get a typical value
        })
    
    
    # make a prediction with our one-row dataframe of user-selected inputs
    # note this produces a numpy array
    prediction = model.predict(user_specified_inputs)
    predicted_nap_duration = prediction[0]   # access the array with the first index slice
    
    # now let's display this on the streamlit app
    st.header('Make a Prediction with the Model!')
    st.markdown("Using the left sidebar, give the model some information about the situation.")
    st.header("The Situation:")
    if user_hour > 12:
        add_clarity = f"({user_hour - 12} pm)"
    else:
        add_clarity = ""
    st.markdown(f"- Hour of the day: {user_hour} {add_clarity}\n - Minutes since previous nursing: {user_min_since_nurse}\n - Minutes since previous nap: {user_min_since_nap}\n - Previous nap duration: {user_prev_nap_duration}")
    st.header("Prediction:")
    st.markdown(f"Baby B's nap will be **{int(predicted_nap_duration)}** minutes long.") # put the result in an f-string


elif app_section == 'Explore the data behind the model':
    # make a header with the title for the app
    st.header('Explore the Baby B Activity Dataset in Tabular Form')
        
    # put the dataframe into the site as a table
    st.write(df)
    
    # import the final naps dataframe that we created elsewhere in a Jupyter notebook
    naps = load_naps_df()

    # add a chart type drop-down
    # a strip chart is like a scatterplot, but one of the axes is categorical
    chart_type = st.sidebar.radio('Choose a chart:', ['Distribution: Nap durations',
                                                      'Time series: Nap duration over time',
                                                      'Relationship: Nap duration vs. hour of the day',
                                                      'Distribution: time since previous nursing',
                                                      'Distribution: time since previous nap'
                                                      ])
        
    # add a chart, based on the selected chart type
    # bar and line charts are built into streamlit, but strip plots aren't, so we'll use plotly
    st.header('Explore the Baby B Activity Dataset Graphically')
    st.markdown("Choose a chart using the left sidebar.")
    
    if chart_type == 'Distribution: Nap durations':
        # make a histogram of time between nap and previous feeding
        #create a plotly figure
        fig = px.histogram(df[df['Activity'] == 'Sleep'], 
                           x = 'Duration (min)',
                           title = "Distribution of Baby B's Nap Durations",
                           labels = {'Duration (min)': 'Nap Duration (Minutes)'},
                           marginal = "box"
                           )
        st.plotly_chart(fig)    # add the figure with this syntax
    
    elif chart_type == 'Time series: Nap duration over time':
        fig = px.scatter(naps, 
                         x = 'Nap Start', 
                         y = 'Nap Duration',
                         title = "Baby B's Nap Durations over Time",
                         labels = {'Duration (min)': 'Nap Duration (Minutes)', 'Nap Start': 'Date and Time'}
                         )
        st.plotly_chart(fig)    # add the figure with this syntax
        st.markdown("Nap durations vary wildly over time,  Note the gap in data in early December.")
    
    elif chart_type == 'Relationship: Nap duration vs. hour of the day':
        fig = px.scatter(naps, 
                         x = 'Hour', 
                         y = 'Nap Duration',
                         title = "Baby B's Nap Duration vs. Hour of the Day",
                         labels = {'Nap Duration': 'Nap Duration (Minutes)'},
                         )
        st.plotly_chart(fig)    # add the figure with this syntax
    
    elif chart_type == 'Distribution: time since previous nursing':
        # make a histogram of time between nap and previous feeding
        #create a plotly figure
        fig = px.histogram(naps,
                           x = 'Minutes since Previous Nursing',
                           title = "Distribution of Time since Previous Nursing",
                           marginal = "box"
                           )
        st.plotly_chart(fig)    # add the figure with this syntax
        st.markdown("Note: large outliers are likely due to data entry issues.  The model is based on decision trees, which are good at handling outlier input values like this.")
    
    elif chart_type == 'Distribution: time since previous nap':
        # make a histogram of time between nap and previous nap
        fig = px.histogram(naps.sort_values(by = 'Minutes since Previous Nap', ascending = False).iloc[1: , :],
                           x = 'Minutes since Previous Nap',
                           title = "Distribution of Time since Previous Nap",
                           marginal = "box"
                           )
        st.plotly_chart(fig)    # add the figure with this syntax
        st.markdown("Note: large outliers and negative values are likely due to data entry issues.  The model is based on decision trees, which are good at handling outlier input values like this.")
    
elif app_section == 'Check out the code on GitHub':
    st.header("Check out the code on my GitHub repository!")
    st.markdown("Everything required to create the model and this web app is [here](https://github.com/jhoernsch/baby-b-nap-predictor) on GitHub")
    
    
# note: the procfile, requirements.txt, and setup.sh are other files that Heroku needs, in addition to this script and the mod.pkl file
# requirements.txt is a file showing version requirements, which is needed for almost anything hosted on a server
# the procfile is giving instructions to Heroku itself.  Basically running Heroku code (start a web app, set it up, then execute this streamlit command)
# setup.sh is a bash script for some of the uploading logistics and configuration stuff
