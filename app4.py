import sklearn 
from sklearn import preprocessing 
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

filename = 'Dec_treev1.sav'
loaded_model = pickle.load(open(filename, 'rb'))

st.set_page_config(layout="wide",page_title="Customer segmentation Dashboard")

def main():
    st.title("Customer Personality Analysis")

    # Add a file uploader
    file =("new_v2df.csv")

    if file is not None:
        # Read the uploaded file into a DataFrame
        df = pd.read_csv(file)

        # Display the DataFrame
        st.write("Uploaded DataFrame:")
        st.write(df)

        # Apply machine learning on the DataFrame
        model = loaded_model

        # Create a new DataFrame for predictions
        predictions_df = pd.DataFrame(columns=['Row', 'Predicted Label'])

        # Iterate over each row of the DataFrame
        for index, row in df.iterrows():
            # Get the features from the row
            features = row[['Education', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'Wines','Fruits', 'Meat', 'Fish', 'Sweets', 'Gold', 'NumDealsPurchases','NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases','NumWebVisitsMonth', 'Age', 'Spent', 'Living_With', 'Children','Family_Size', 'Is_Parent']].values.reshape(1, -1)  # Replace with your actual feature column names

            # Make predictions using the machine learning model
            predictions = model.predict(features)

            # Add the predictions to the predictions DataFrame
            predictions_df = predictions_df.append({'Row': index + 1, 'Predicted Label': predictions[0]}, ignore_index=True)

        # Display the predictions DataFrame as a table
        st.write("Predictions:")
        st.dataframe(predictions_df)

        # Extract the predicted labels from the DataFrame
        predicted_labels = predictions_df['Predicted Label']

         # Calculate the total predictions for each label
        label_counts = predicted_labels.value_counts().reset_index()
        label_counts.columns = ['Label', 'Total Predictions']

        # Create a bar chart to visualize the label counts
        fig = px.bar(label_counts, x='Label', y='Total Predictions', color='Label')

        # Set plot layout
        fig.update_layout(
            title='Total Predictions per Label',
            xaxis_title='Label',
            yaxis_title='Total Predictions'
        )

        # Display the chart
        st.plotly_chart(fig)
        # Display the label counts as a table
        st.write("Total Predictions per Label:")
        st.write(label_counts)

        
        

