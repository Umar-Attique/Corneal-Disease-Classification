import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from PIL import Image
import io
import base64

# Step 1: Save your trained model
MODEL_PATH = r"C:\Users\Umar Attique\Downloads\corn_disease.keras"

# Load the model
model = load_model(MODEL_PATH)

# Class names (update according to your dataset)
classes = ['Blight', 'Common_Rust', 'Gray_leaf_spot', 'Healthy']

# Create the Dash App
app = dash.Dash(__name__)
app.title = 'Corn Disease Classification'  # Tab title

app.layout = html.Div(style={'backgroundColor': '#f8f9fa', 'padding': '20px'}, children=[
    # Display title on the page
    html.H1("Corn Disease Classification", style={
        "textAlign": "center",
        "color": "#007bff",
        "marginBottom": "20px"
    }),

    # Image upload and preview section
    html.Div(style={
        'border': '2px solid #007bff',
        'borderRadius': '10px',
        'padding': '20px',
        'boxShadow': '0px 4px 10px rgba(0, 0, 0, 0.1)',
        "textAlign": "center",
        "maxWidth": "500px",
        "margin": "0 auto"
    }, children=[
        dcc.Upload(
            id="upload-image",
            children=html.Div([
                "Drag and Drop or ", html.A("Select an Image", style={'color': '#007bff', 'fontWeight': 'bold'})
            ]),
            style={
                "width": "80%",
                "height": "40px",
                "lineHeight": "40px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "10px",
                "textAlign": "center",
                "margin": "0 auto",
                "backgroundColor": "#e9ecef",
                "color": "#6c757d"
            },
            accept=".jpg,.jpeg,.png"
        ),
        html.Div(id="preview-image", style={"textAlign": "center", "marginTop": "20px"}),
    ]),

    # Submit button outside the boundary
    html.Div(style={"textAlign": "center", "marginTop": "20px"}, children=[
        html.Button(
            "Prediction",
            id="submit-button",
            n_clicks=0,
            style={
                "padding": "10px 20px",
                "backgroundColor": "#007bff",
                "color": "white",
                "border": "none",
                "borderRadius": "5px",
                "cursor": "pointer",
                "fontSize": "16px",
                "fontWeight": "bold"
            },
        ),
    ]),

    # Output section with loading spinner
    dcc.Loading(
        id="loading",
        type="circle",  # Circle spinner
        children=[
            html.Div(id="output-image-upload", style={"textAlign": "center", "marginTop": "30px"}),
        ]
    ),
])


# Preprocess the uploaded image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0  # Normalize
    return image_array

# Callback to handle image preview and upload
@app.callback(
    Output("preview-image", "children"),
    [Input("upload-image", "contents")],
)
def preview_uploaded_image(contents):
    if contents is not None:
        return html.Img(src=contents, style={"height": "200px", "marginTop": "10px", "borderRadius": "10px"})
    return ""

# Callback to handle predictions
@app.callback(
    Output("output-image-upload", "children"),
    [Input("submit-button", "n_clicks")],
    [State("upload-image", "contents"), State("upload-image", "filename")]
)
def update_output(n_clicks, contents, filename):
    if n_clicks > 0 and contents is not None:
        # Decode the image
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))

        # Preprocess the image
        processed_image = preprocess_image(image, (224, 224))

        # Predict using the model
        predictions = model.predict(processed_image)[0]
        predicted_class = classes[np.argmax(predictions)]
        predicted_prob = predictions[np.argmax(predictions)] * 100

        # Create pie chart
        fig = go.Figure(data=[
            go.Pie(labels=classes, values=predictions, hole=0.4)
        ])
        fig.update_layout(
    title={
        "text": "Class Probabilities",
        "font": {
            "size": 20,  # Set the font size
            "family": "Arial",  # Optionally set font family
            "color": "black"  # Set the title color here
        }
    },
    height=400
)

        # Display predictions and chart
        return html.Div([
            html.H3(f"Predicted Disease : {predicted_class}", style={"marginTop": "20px", "color": "#343a40"}),
            html.H4(f"Probability: {predicted_prob:.2f}%", style={"color": "#17a2b8"}),
            dcc.Graph(figure=fig)
        ])

    return ""  # Empty if no content

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
