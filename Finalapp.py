from sklearn import preprocessing 
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
import requests
import pandas as pd

st.set_page_config(layout="wide",page_title="Customer segmentation Dashboard")
st.title("Customer Personality Analysis")
###############################
# Add a File Uploader widget
filess = st.file_uploader("Upload a CSV file")

if filess is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(filess)

    # Display the uploaded data
    st.write("Uploaded CSV data:")
    st.write(df)

# GitHub raw URL for the file
github_raw_url = "https://raw.githubusercontent.com/Rohit2350/Customer_Segmentation_Dashboard/main/new_v2df.csv"

try:
    # Fetch and display the CSV file from GitHub
    file= pd.read_csv(github_raw_url)
    st.write("Data fetched from GitHub:")
    st.write(file)
except Exception as e:
    st.write(f"Failed to fetch data from GitHub: {e}")

###############################3
filename = 'Dec_treev1.sav'
loaded_model = pickle.load(open(filename, 'rb'))

def main():    
    # Add a file uploader
    #file = st.file_uploader("Upload a CSV file", type="csv")

    if file is not None:
        # Read the uploaded file into a DataFrame
        #df = pd.read_csv(file)

        # Display the DataFrame
        #st.write("Uploaded DataFrame:")
        #st.write(df)

        # Apply machine learning on the DataFrame
        model = loaded_model

        # Create a new DataFrame for predictions
        predictions_df = pd.DataFrame(columns=['Row', 'Predicted Label'])

        # Iterate over each row of the DataFrame
        for index, row in file.iterrows():
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

        
        


if __name__ == '__main__':
    main()