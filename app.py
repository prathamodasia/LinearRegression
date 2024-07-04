import streamlit as st
import pandas as pd
import numpy as np
import pickle

#load the model
clf=pickle.load(open("mymodel.pkl","rb"))
def predict(data):
    clf=pickle.load(open("mymodel.pkl","rb"))
    return clf.predict(data)

st.title("Advertising spends prediction using Machine Learning")
st.markdown("This model identify total spends on Advertising")

st.header("Advertising spend on various media")
col1,col2=st.columns(2)

with col1:
    st.text("TV")
    tv=st.slider("Adver. spends on TV",1.0,10000.0,0.5)
    st.text("Radio")
    rd=st.slider("Adver. spends on Radio",1.0,10000.0,0.5)
    st.text("Newspaper")
    newspaper=st.slider("Adver. spends on Newspaper",1.0,10000.0,0.5)
    
st.text('')
if st.button("sales Prediction"):
    result=clf.predict(np.array([[tv,rd,newspaper]]))
    st.text(result[0])
    
st.markdown("Developed by Pratha Modasia at NIELIT Daman")