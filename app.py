import streamlit as st
import numpy as np
import pandas as pd
import joblib


# Lets load the joblib instances over here
with open('pipeline.joblib','rb') as file:
    preprocess = joblib.load(file)

with open('model.joblib','rb') as file:
    model = joblib.load(file)

# Lets take the inputs from the user
st.title('HELP NGO ORGANIZATION')
st.subheader('This Application will help to identify the Development category of the country using Socio-Economic factors. The Original data has been Clustered using K-Means')

# Lets take the inputs
gpp = st.number_input('Enter the GPPP of a count(GDP per population)')
income = st.number_input('Enter Income per population')
imports = st.number_input('Imports of goods and services per capita. Given as %age of the GDP per capita')
exports = st.number_input('Exports of goods and services per capita. Given as %age of the GDP per capita')
inflation = st.number_input('INFLATION : The measurement of the annual growth rate of the Total GDP')
lf_expcy = st.number_input('LIFE EXPECTANCY : The average number of years a new born child would live if the current mortality patterns are to remain the same')
fert = st.number_input('FERTILITY : The number of children that would be born to each woman if the current age-fertility rates remain the same.')
health = st.number_input('Total health spending per capita. Given as %age of GDP per capita')
child_mort = st.number_input('CHILD MORTALITY : Death of children under 5 years of age per 1000 live births')

input_list = [child_mort,exports,health,imports,income,inflation,lf_expcy,fert,gpp]
final_input_list = preprocess.transform([input_list])
if st.button('Predict'):
    prediction = model.predict(final_input_list)[0]
    if prediction==0:
        st.success('DEVELOPING')
    elif prediction==1:
        st.success('DEVELOPED')
    else:
        st.error('UNDERDEVELOPED')


# success show gren box and error shows red box
