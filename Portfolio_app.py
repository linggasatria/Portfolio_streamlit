import streamlit as st

st.set_page_config(layout="wide")

#page setup
About_Me_Page = st.Page(
    page = "page_views/about_me.py",
    title = "About Me",
    
    default = True
)

Project_Overview = st.Page(
    page = "page_views/project_overview.py",
    title = "Project Overview",
    
)

Visualisasi_Model = st.Page(
    page = "page_views/visualisasi_model.py",
    title = "Visualisasi Model",
    
   
)

Machine_learning = st.Page(
    page = "page_views/machine_learning.py",
    title = "Machine Learning",
    
)

Recomendation_system = st.Page(
    page = "page_views/Recomended_system.py",
    title = "Recomendation System",
    
)

#navigation
pg = st.navigation(
    pages = [About_Me_Page, Project_Overview, Visualisasi_Model, Machine_learning, Recomendation_system]
)

st.logo("assets/icon.png")

pg.run()