import streamlit as st
import os
import tensorflow as tf
import pandas as pd
import plotly.express as px

# Page Title
st.title("ASL Model Performance Dashboard")
st.write("📊 A dashboard to showcase model performance statistics 📊 ")

MODEL_FOLDER = "models"

# Get list of available models in the folder
model_files = [f for f in os.listdir(MODEL_FOLDER)]

# Streamlit dropdown to select a model
selected_model = st.selectbox("Select a Model:", model_files)

# Load the selected model when user selects
if selected_model:
    model_path = os.path.join(MODEL_FOLDER, selected_model + "/" + selected_model + ".h5")
    model = tf.keras.models.load_model(model_path)
    
    if not model:
        st.error("Model could not be loaded")
    else:
        st.success(f"Loaded Model: {selected_model}")

        path = os.path.join(MODEL_FOLDER, selected_model)
        history = pd.read_json(path + "/history.json")
        metrics = pd.read_json(path + "/metrics.json")
        matrix = pd.read_json(path + "/matrix.json")

        st.header("History")
        st.subheader("🔍 Raw Data")
        st.write(history.sort_values(by="accuracy", ascending=False))
        
        st.subheader("Accuracy during training (30 epochs)")
        fig = px.line(
            history,
            x=history.index,  # Use epoch as x-axis
            y="accuracy",
            markers=True,  # Enable markers
            title="Model Accuracy Over Epochs",
            labels={"accuracy": "Accuracy", "epoch": "Epoch"}
        )

        # Adjust Y-axis limits for zoom effect
        fig.update_yaxes(range=[history["accuracy"].min() - 0.01, history["accuracy"].max() + 0.01])

        # Show Plot in Streamlit
        st.plotly_chart(fig)

        st.header("Performance Metrics")
        st.subheader("🔍 Raw Data")

        st.table(metrics)

        options = ["accuracy", "label"]
        sorter = st.selectbox(options=options, label="Select option to sort by:")
        
        metrics = metrics.sort_values(by=sorter, ascending=(sorter == "label"))

        # Plotly Bar Chart
        fig = px.bar(
            metrics,
            x="label",
            y="accuracy",
            color="accuracy",
            color_continuous_scale="viridis",  # Color gradient
            labels={"accuracy": "Accuracy", "label": "Labels"},
            title="Model Accuracy by Label"
        )

        # Customize layout
        fig.update_layout(
            xaxis_tickangle=0,
            yaxis=dict(title="Accuracy (%)"),
            coloraxis_colorbar=dict(title="Accuracy")
        )

        # Show Chart
        st.plotly_chart(fig)
        

        st.header("Results Matrix")
        
    
        fig = px.imshow(
            matrix,
            color_continuous_scale='Viridis',  # You can change the color scale
            title="Heatmap",
            labels=dict(x="Ground Truth", y="Model Guesses", color="Intensity")
        )
        fig.update_layout(
            width=1000,  # Set the width (in pixels)
            height=1000,  # Set the height (in pixels)
        )

        # Display the heatmap in Streamlit
        st.plotly_chart(fig)
                
        st.table(matrix)