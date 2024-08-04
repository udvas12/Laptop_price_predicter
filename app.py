import streamlit as st
import pickle
import numpy as np
import pandas as pd
df = pickle.load(open('df.pkl','rb'))
pipe=pickle.load(open('pipe.pkl','rb'))
st.title('Laptop Predictor:-')
# Brand

company = st.selectbox('Brand',df['Company'].unique())

# Type of Laptop

type_name = st.selectbox('Type',df['TypeName'].unique())

# Ram

ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# Weight

weight = st.number_input('Weight of the Laptop:')

# Touchscreen

touchscreen = st.selectbox('Touchscreen',['YES','NO'])

# IPS Display

ips = st.selectbox('IPS Display',['YES','NO'])

# resolution

resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# Screen Size

screen_size = st.number_input("Screen Size")

# CPU

cpu = st.selectbox('CPU',df['Cpu brand'].unique())

# HDD

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

# SSD

ssd=st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

# GPU

gpu = st.selectbox('GPU',df['Gpu_brand'].unique())

# OS

os = st.selectbox('OS',df['os'].unique())

if st.button('Predicted price'):
    if touchscreen == 'YES':
        touchscreen = 1
    else:
        touchscreen = 0

    if (ips=='YES'):
        ips=1
    else:
        ips=0
    x_res=int(resolution.split('x')[0])
    y_res=int(resolution.split('x')[1])
    ppi=((x_res**2)+(y_res**2))**0.5/screen_size
    query=np.array([company,type_name,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query=query.reshape(1,12)
    st.title(np.exp(pipe.predict(query)))
