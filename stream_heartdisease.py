

from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('final_knn_model')






def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('main.jpg')
    image_office = Image.open('side.jpg')
    st.image(image,use_column_width=True)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.sidebar.info('This app is created to detect whether a person has heart disease')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_office)
    st.title("Predicting heart disease")
    if add_selectbox == 'Online':
        age=st.number_input('age' , min_value=1, max_value=80, value=1)
        sex = st.selectbox('Sex', ['0','1'])
        cp = st.selectbox('chest pain type (4 values)', ['0','1','2','3'])
        trestbps =st.number_input('resting blood pressure',min_value=1, max_value=200, value=1)
        chol= st.number_input('serum cholesterol in mg/dl', min_value=10, max_value=600, value=10)
        fbs = st.selectbox('fasting blood sugar > 120 mg/dl', ['0','1'])
        restecg=st.selectbox('resting electrocardiographic results (values 0,1,2)', ['0','1','2'])
        thalach=st.number_input('maximum heart rate achieved', min_value=10, max_value=300, value=10)
        exang=st.selectbox('9.	exercise induced angina', ['0','1'])
        oldpeak=st.number_input('oldpeak = ST depression induced by exercise relative to rest', min_value=0.0, max_value=8.0, value=0.1)
        slope=st.selectbox('the slope of the peak exercise ST segment', ['0','1','2'])
        ca=st.number_input('number of major vessels (0-3) colored by flourosopy', min_value=0, max_value=10, value=1)
        thal=st.selectbox('thal: 3 = normal; 6 = fixed defect; 7 = reversable defect', ['0','1','2','3'])
        output=""
        input_dict={'age':age,'sex':sex,'cp':cp,'trestbps':trestbps,'chol':chol,'fbs':fbs,'restecg':restecg,'thalach':thalach,'exang':exang,'oldpeak':oldpeak,'slope':slope,'ca':ca,'thal':thal}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('The output is {}'.format(output))
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
def main():
    run()

if __name__ == "__main__":
  main()
