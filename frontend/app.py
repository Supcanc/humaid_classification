import streamlit as st
import requests
from config import PREDICT_URL, ID2LABEL

st.title('Humanitarian aid tweets classification')

text = st.text_area('Write down your text for classification here:')

if st.button('Classify'):
    response = requests.post(PREDICT_URL, json={'string': text}).json()
    st.write(f'Predicted class for your input - "{response['class_labels'][0]}"')
    probs = response['classes_probs'][0]
    label_names = list(ID2LABEL.values())
    classes_percentages = {label_names[i]: probs[i] * 100 for i in range(len(probs))}
    st.bar_chart(
        classes_percentages,
        x_label='Classes names',
        y_label='Confidence scores(%)',
        horizontal=True
    )