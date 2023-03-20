#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:55:23 2023

@author: anualao
"""

#importing dependencies
import numpy as np
import pandas as pd
from joblib import load
import streamlit as st

# loading in the data(The dumped model)
model = load('Iris_model.joblib')


#Backend
def predictions(sepallength, sepalwidth, petallength, petalwidth):
    prediction = model.predict(np.array([[sepallength, sepalwidth, petallength, petalwidth]]))
    
 
    return prediction



# fuction to create the UI
def main():
    st.title('Iris flower model')
    st.header('Iris Dataset')
    
    

    sepallength = st.number_input('Enter sepal length: ')
    sepalwidth = st.number_input('Enter sepal width: ')
    petallength = st.number_input('Enter petal length: ')
    petalwidth = st.number_input('Enter petal width: ')
    
    
    button = st.button('predict')
    
    result = ''
    
    
    if (button):
        result = predictions(sepallength, sepalwidth, petallength, petalwidth)
        st.write('predicting...')
        if result == 0:
            st.success('This is a Setosa')
        elif result == 1:
            st.success('This is a Versicolor')
        elif result == 2:
            st.success('This is Virginica')
            
        else:
            st.success('This flower does not exist')
        
        
    
    
    
    
if __name__== '__main__':
    main()