import streamlit as st
import numpy as np
import pandas as pd

st.title("Dominant Color Extraction")
st.subheader("Input Image")
img= st.file_uploader("Choose an image")
if img is not None: 
    st.header("Orignal Image")
    st.image(img) # display the uploaded image

    from matplotlib import pyplot as plt 
    img=plt.imread(img)
    n = img.shape[0]*img.shape[1] #len x breadth
    all_pixels = img.reshape((n, 3)) 
    from sklearn.cluster import KMeans
    model  = KMeans (n_clusters = 3) #5 dominant colors will be the centre for the clusters
    model.fit(all_pixels)
    centers = model.cluster_centers_.astype('uint8')
    new = np.zeros((n, 3), dtype='uint8') 
    for i in range(n): #iteration for each pixel
        group_idx = model.labels_[i] #get group/cluster of 1st pixel for eg
        new[i] = centers[group_idx]
    new = new.reshape(*img.shape)

    st.header("Modified Image")
    st.image(new)
