import streamlit as st

# Headings
st.markdown("# Welcome! Let's predict the likehood of someone being a LinkedIn User")
st.markdown("#### Use this application to determine the likelihood of an individual being a LinkedIn user based on certain criteria")
st.markdown("_Created by Michaela Caudill_")


# Getting input for "income","education","parent","married","female","age"
st.markdown("##### First, we'll need some information...")
income = st.selectbox("Income (household)", 
               options = ["Less than $10,000",
                          "10 to under $20,000",
                          "20 to under $30,000",
                          "30 to under $40,000",
                          "40 to under $50,000",
                          "50 to under $75,000",
                          "75 to under $100,000",
                          "100 to under $150,000",
                          "$150,000 or more"])

education = st.selectbox("Highest level of school/degree completed", 
               options = ["Less than high school (Grades 1-8 or no formal schooling)",
                          "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
                          "High school graduate (Grade 12 with diploma or GED certificate)",
                          "Some college, no degree (includes some community college)",
                          "Two-year associate degree from a college or university",
                          "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
                          "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
                          "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"])

parent = st.selectbox("Are you a parent of a child under 18 living in your home?", 
               options = ["Yes, I have a child under 18 living my home",
                          "No, I do not have a child under 18 living my home"])

married = st.selectbox("Are you currently married?", 
               options = ["Yes, I am currently married",
                          "No, I am not currently married"])

female = st.selectbox("Do you identify as a female?", 
               options = ["Yes, I identify as a female",
                          "No, I do not identify as a female"])

age = st.slider(label="Enter your age", 
           min_value=1,
           max_value=98,
           value=49)


# Creating labels for inputs

## income
if income == "Less than $10,000": income = 1
elif income == "10 to under $20,000": income = 2
elif income == "20 to under $30,000": income = 3
elif income == "30 to under $40,000": income = 4
elif income == "40 to under $50,000": income = 5
elif income == "50 to under $75,000": income = 6
elif income == "75 to under $100,000": income = 7
elif income == "100 to under $150,000": income = 8
else: income = 9

## eduction
if education == "Less than high school (Grades 1-8 or no formal schooling)": education = 1
elif education == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)": education = 2
elif education == "High school graduate (Grade 12 with diploma or GED certificate)": education = 3
elif education == "Some college, no degree (includes some community college)": education = 4
elif education == "Two-year associate degree from a college or university": education = 5
elif education == "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)": education= 6
elif education == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)": education = 7
else: education = 8
    
## parent
if parent == "Yes, I have a child under 18 living my home": parent = 1
else: parent = 0

## married
if married == "Yes, I am currently married": married = 1
else: married = 0

## gender
if female == "Yes, I identify as a female": female = 1
else: female = 0

## printing outcome with variables that'll go in the model
st.markdown("##### Thanks for that information! The below inputs will be put into the model")
st.write(f"Income (household): {income} | Education level: {education} | Parent): {parent} | Married: {married} | Female: {female} | Age: {age}")
st.markdown("*Note on the output: 0=not LinkedIn user, 1= LinkedIn user*")

# Code for the app/model 
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    x = np.where(x == 1,
            1, 0)
    return x
    
ss = pd.DataFrame({
    "sm_li":clean_sm(s["web1h"]), 
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] == 1, 1, 0),
    "married":np.where(s["marital"] == 1, 1, 0),
    "female":np.where(s["gender"] == 1, 1, 0),
    "age":np.where(s["age"] > 98, np.nan, s["age"])
    })

ss = ss.dropna()
ss['age'] = ss['age'].astype(int) # update age from float to int

y = ss["sm_li"]
X = ss[["income","education","parent","married","female","age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility
# Initialize algorithm 
lr = LogisticRegression()
# Fit algorithm to training data
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

# Defining function to take inputs 
def linkedin_app(income, education, parent, married, female, age):
    # New data for features: "income","education","parent","married","female","age"
    person = [income, education, parent, married, female, age]
    person_predicted = lr.predict([person])
    person_probs = lr.predict_proba([person])

    # 0=not LinkedIn user, 1= LinkedIn user

    return st.markdown(f"### This person's predicted class is {person_predicted[0]}. The probability this person is a LinkedIn user is {person_probs[0][1]}")


linkedin_app(income, education, parent, married, female, age)


