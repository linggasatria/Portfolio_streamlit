import streamlit as st

#kolom 1
col_photo, col_about = st.columns([1,3], gap = "medium")

with col_photo :
    st.image("assets/poto_profile.png")
    
with col_about:
    st.title("Lingga Satria Permana")
    st.write("""
            I am a Design Specialist and have been working in the Advertising Company for more than 8 years. I have professional skills and extensive experiences in the creative field, especially in making visual output both digital and printed. I have contributed in the marketing projects of more than a hundred clients, mostly from multinational companies. 

I have been taking intensive course of AI and Machine Learning in Dibimbing.id since April 2025. Because recently i am eager to learn about AI and Machine Learning that give big impact in the world of technology and i believe it can makes every business rapidly growing. 

I am a proactive and solution-oriented individual, with a high learning spirit and the ability to adapt in a dynamic work environment. I can build positive and collaborative working relationships. With a detailed and analytical approach, I always try to provide the best work results and improve efficiency in every task. I believe that professional growth comes from challenges, and I am always ready to face and overcome them with the right strategy.
            """)


#Kolom 2
col_edu, col_exp, col_skill = st.columns (3, gap="medium")

with col_edu :
    st.subheader("Education")
    st.write("""
         - AI & Machine Learning Engineer, Dibimbing.id,
         Apr - Present
         - BDes in Visual Communication Design (DKV), UNIKOM, GPA : 3.19/4.00
         2010 - 2015
         """)
    
with col_exp :
    st.subheader("Experience")
    st.write("""
         _DESIGN SPECIALIST, PT. PANON MAHIA NUSA Feb 2017 – Present_
         - Creating and developing visual concepts
         - Collaborating with the marketing team to finalize the visual concepts and prepare marketing tools
         - Creating the rebranding concept and strategy
         - Creating templates and asset designs
         - Creating final artwork when the design has reached final approval
         """)
    
with col_skill :
    st.subheader("Skills")
    st.write("""
         - Data Science Fundamentals : Data Manipulation, Data Cleaning, Data Distribution, EDA (Exploratory Data Analysis), Data Visualization
         - Machine Learning : Model Supervised (Simple & Multiple Regression), Regularized Regression, Clustering Techniques, Hyperparameter Tuning
         - Advanced AI & Frameworks : NLP (Natural Language Processing), Computer Vision, Generative AI, RAG (Retrieval-Augmented Generation), LangChain Framework
         - MLOps : MLOps
         - Databases : PosgreSQL
         """)
    
#Project