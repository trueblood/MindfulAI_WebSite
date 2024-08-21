# -*- coding: utf-8 -*-
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import random
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import os
import requests
import os

BASE_URL = os.environ.get("BASE_URL")
api_key = os.environ.get("API_KEY")

def insert_emotion_cnn_ai_training(data):
    url = f"{BASE_URL}/api/insert_emotion_cnn_ai_training"
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    return response.json()

def insert_encoding_model_training(data):
    url = f"{BASE_URL}/api/insert_encoding_model_training"
    response = requests.post(url, json=data, headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'})
    return response.json()

def get_feedback_by_user(user_id):
    url = f"{BASE_URL}/api/get_feedback_by_user/{user_id}"
    response = requests.get(url, headers={'Authorization': f'Bearer {api_key}'})
    return response.json()

def get_feedback_rating_options():
    url = f"{BASE_URL}/api/get_feedback_rating_options"
    response = requests.get(url, headers={'Authorization': f'Bearer {api_key}'})
    return response.json()

def get_rl_training_info():
    url = f"{BASE_URL}/api/get_rl_training_info"
    response = requests.get(url, headers={'Authorization': f'Bearer {api_key}'})
    return response.json()

def insert_rl_training_info(data):
    url = f"{BASE_URL}/api/insert_rl_training_info"
    response = requests.post(url, json=data, headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'})
    return response.json()


# Sample Data Preparation
np.random.seed(42)
timestamps = pd.date_range(end=datetime.datetime.now(), periods=60, freq='1T')  # 60 minutes range
feedback_scores = np.random.randint(1, 6, size=60)  # Feedback scores between 1 and 5
emotions = ['Happy', 'Sad', 'Angry']
locations = ['Commercial', 'Residential']

data = pd.DataFrame({
    'Timestamp': timestamps,
    'Feedback_Score': feedback_scores,
    'Emotion': np.random.choice(emotions, size=60),
    'Location': np.random.choice(locations, size=60)
})

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.YETI, dbc.icons.FONT_AWESOME],
)

# List of emotions and corresponding image files (example filenames)
emotions = [
    {"name": "Happy", "image": "images/happy/happy_00b597a317f73e5832275a0a5f9aa8250f1a4450bd55ac20387f2c9d.jpg", "location": "Commercial"},
    {"name": "Sad", "image": "images/sad/sad_01b1763812bc6d9932343b0122aefff73ed1a0cce2f252f8b3a80546.jpg", "location": "Residential"},
    {"name": "Angry", "image": "images/angry/angry_0b4fb5f008ae748b2bfe8fc4d35af1d37ab56120acbbd135be98fdb5.jpg", "location": "Commercial"}
]

# Sample data for the table
leaderboard_data = {
    "Exercise": [
        "Journaling to process thoughts and reflect on emotions.",
        "Writing down daily goals to organize and focus the mind.",
        "Breathing in deeply through the nose, hold for a few seconds, then exhale slowly.",
        "Writing a gratitude list to enhance positive emotions.",
        "Practice mindful breathing by counting each breath cycle.",
        "Guided visualization breathing exercises for stress relief.",
        "Structured journaling to explore thought patterns.",
        "Perform the downward dog yoga pose for a quick energy boost.",
        "Stretching routine with focus on controlled breathing.",
        "Free-form journaling to let thoughts flow without restrictions."
    ],
    "Emotion": ["Happy", "Sad", "Sad", "Sad", "Angry", "Sad", "Sad", "Angry", "Sad", "Sad"],
    "Exercise Type": ["Journaling", "Journaling", "Breathing Exercise", "Journaling", "Breathing Exercise", "Breathing Exercise", "Journaling", "Yoga", "Yoga", "Journaling"],
    "Location": ["Commercial", "Residential", "Residential", "Residential", "Commercial", "Commercial", "Residential", "Commercial", "Residential", "Residential"],
    "Q-Value": [0.88, 0.83, 0.72, 0.79, 0.74, 0.69, 0.77, 0.68, 0.65, 0.70],
    "Source": ["user_feedback_for_that_user", "recommended_table", "recommended_table", "user_feedback_for_that_user", "user_feedback_for_that_user", "user_feedback_for_that_user", "recommended_table", "user_feedback_for_that_user", "recommended_table", "recommended_table"]
}

# Convert the data into a Pandas DataFrame
df_leaderboard = pd.DataFrame(leaderboard_data)

#  make dataframe from  spreadsheet:
df = pd.read_csv("assets/historic.csv")

MAX_YR = df.Year.max()
MIN_YR = df.Year.min()
START_YR = 2007

# since data is as of year end, need to add start year
df = (
    df._append({"Year": MIN_YR - 1}, ignore_index=True)
    .sort_values("Year", ignore_index=True)
    .fillna(0)
)

COLORS = {
    "background": "whitesmoke",
}

# Markdown component to display the selected exercise
asset_allocation_text = dcc.Markdown(id="asset_allocation_text")
asset_allocation_card = dbc.Card(
    [
        html.Br(),
        html.H5(
            "AI Suggested Mindfulness Exercise By Selected Emotion Card",
            className="card-title  text-center"
        ),
        html.Br(),
        html.P(asset_allocation_text, className="card-text text-center"),
    ],
    className="mt-2",
)

learn_text = dcc.Markdown(
    """
    ## System Overview

    Our system captures users' emotional expressions from facial images and location data. These are processed through a Convolutional Neural Network (CNN) and a Text Embedding Model to generate vector embeddings. These vectors are then sent to our API and stored in the **TiDB Serverless Database**.

    #### Showcasing Data Types and Benefits

    Our application stores a variety of data types in TiDB, from user emotional expressions to exercise feedback. TiDB Serverless supports our need for quick, efficient, and cost-effective data handling, allowing us to focus on enhancing user experience and expanding our service capabilities without compromising performance.
    
    ## Reinforcement Learning Model

    Our Reinforcement Learning (RL) model employs a hybrid of **on-policy** and **off-policy** methods. It activates and selects the optimal mindfulness exercise by integrating current state assessments with historical data and vector semantic search queries within TiDB.

    ### Q-Learning with TiDB Integration

    The model utilizes **Q-Learning**, an algorithm designed to predict the best action in a given situation to maximize reward. Feedback from users on the suggested exercises is used to adjust the Q-values in the TiDB Serverless Database, refining future recommendations. If no feedback is received, the Q-value remains unchanged.

    #### Our Policy Using TiDB Vector Semantic Search

    We employ TiDB Vector Semantic Search for its ability to handle high data volumes and for fast query data retrieval, which ensures that our agent swiftly identifies the best possible action for each user scenario.

     #### Semantic Search in TiDB Serverless

    TiDB Serverless enhances our application by offering quick scalability and responsive query times, crucial for handling AI workloads and traditional data processing tasks alike. This makes it an indispensable part of our AI-driven application.

    ---

    #### What is a Reinforcement Learning Model?

    A reinforcement learning model involves training an agent (in our case, the system itself) to make sequences of decisions. The agent learns to achieve a goal in an uncertain, potentially complex environment. In our project, the agent decides the most suitable mindfulness exercise based on the user's current emotional state.

    #### What is a CNN Model?

    A Convolutional Neural Network (CNN) is a type of deep learning algorithm primarily used for processing data with a grid-like topology, such as images. CNNs excel in tasks like image recognition and classification by extracting features through layers of filters and pooling operations, enabling them to identify patterns effectively.

    #### What is a Text Embedding Model?

    A Text Embedding Model transforms text into numerical vectors that capture semantic meaning. Commonly used in natural language processing (NLP), these models enable machines to perform complex text-related tasks like sentiment analysis and text classification by understanding context and relationships within the text.
    
    #### Vectors, Vector Index, and Vector Embeddings

    Vectors are numerical representations of data, which we use extensively in our project. We store these vectors in TiDB, utilizing its vector indexing capabilities to efficiently perform semantic searches that match users' emotional states with appropriate mindfulness exercises.
    
    #### What is Semantic Vector Search?

    Semantic Vector Search uses vector embeddings to enhance search accuracy by finding content based on semantic similarities. This approach allows for more relevant results by understanding the underlying meanings in the data.

    #### Additional Resources

    - [Download the Workflow PDF](/assets/Documents_EmbraceAIWorkFlow.pdf) for a detailed overview of our AI swarm process along with how we integrate TiDB into our solution.
    - Explore our GitHub repositories to see the code and contribute to the projects:
    - [Embrace AI Website](https://github.com/trueblood/MindfulAI_WebSite)
    - [Embrace CNN](https://github.com/trueblood/EmbracePath_AI)
    - [TiDB Data Assistant API](https://github.com/trueblood/TiDB_Data_Assistant)
    - [Embrace Dog Wagon RL AI](https://github.com/trueblood/Dog_Wagon_AI)
    """
)

footer = html.Div(
    dcc.Markdown(
        """
         This information is intended solely as general information for educational
        and entertainment purposes only.
        """
    ),
    className="p-2 mt-5 bg-primary text-white small",
)

feedback_leaderboard_table = dbc.Card(
    [
        html.H4("Feedback Leaderboard", className="card-title text-center mt-3"),
        dash_table.DataTable(
            id='leaderboard_table',
            columns=[{"name": col, "id": col} for col in df_leaderboard.columns],
            data=df_leaderboard.to_dict('records'),
            style_header={
                'backgroundColor': 'rgb(255, 255, 255)',
                'color': 'black',
                'border': '1px solid black',
                'textAlign': 'center',  # Center-align the header text
                'fontSize': '14px',  # Slightly larger font size for headers
            },
            style_cell={
                'textAlign': 'center',  # Center-align the text
                'padding': '5px',  # For a compressed look
                'whiteSpace': 'normal',
                'height': 'auto',
                'overflow': 'hidden',
                'fontSize': '12px',  # Reduce the font size
            },
            style_table={
                'height': '300px',  # Adjust the height to trigger scroll
                'overflowY': 'auto',
                'border': '1px solid black',
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Source} = "user_feedback_for_that_user"'},
                    'backgroundColor': 'rgb(220, 220, 220)'
                },
                {
                    'if': {'filter_query': '{Source} = "recommended_table"'},
                    'backgroundColor': 'rgb(245, 245, 245)'
                }
            ],
            page_size=10
        ),
    ],
    body=True,
    className="mt-4",
    style={
        'backgroundColor': 'white',
        'border': '1px solid black',
        'color': 'black',
    }
)

mindfulness_feedback_scale = [
        {
        "label": "5: Very Helpful - Exercise perfectly suited for the situation and fully relieved stress",
        "start_yr": 5,
        "description": "The exercise was highly effective and perfectly aligned with my needs."
    },
        {
        "label": "4: Helpful - Exercise significantly improved emotional state",
        "start_yr": 4,
        "description": "The exercise was effective in helping manage emotions with good results."
    },
        {
        "label": "3: Moderately Helpful - Exercise helped manage some emotions but not entirely effective",
        "start_yr": 3,
        "description": "The exercise provided moderate relief but wasn’t fully sufficient."
    },
        {
        "label": "2: Slightly Helpful - Some minor impact but overall felt irrelevant",
        "start_yr": 2,
        "description": "The exercise had a small positive effect, but was mostly unhelpful."
    },
    {
        "label": "1: Not Helpful - Exercise felt overwhelming or not useful in managing emotions",
        "start_yr": 1,
        "description": "The exercise did not provide relief or made the situation worse."
    },
    {
        "label": "Exercise did not apply to my situation - The exercise didn’t fit my current emotional context",
        "start_yr": 0,
        "description": "The exercise wasn’t relevant to what I was experiencing."
    }
]

time_period_card = dbc.Card(
    [
        html.H5(
            "Suggested Mindfulness Exercise Feedback Rating",
            className="card-title text-center",
        ),
        html.Hr(),
        dbc.RadioItems(
            id="time_period",
            options=[
                {"label": period["label"], "value": i}
                for i, period in enumerate(mindfulness_feedback_scale)
            ],
            value=5,
            labelClassName="mb-2",
        ),
    ],
    body=True,
    className="mt-4",
)

emotion_cards = html.Div(
    dbc.Row(
        [
            dbc.Col(
                dbc.Button(
                    dbc.Card(
                        [
                            dbc.CardImg(
                                src=app.get_asset_url(emotion['image']), 
                                top=True, 
                                style={"padding": "10px", "filter": "grayscale(100%)"}  # Grayscale filter
                            ),
                            dbc.CardBody([
                                html.Hr(),
                                html.P(
                                    [
                                        "Click on card to classify emotion."  # Second line
                                    ],
                                    className="card-text center-text",
                                ),
                                html.P(
                                    [
                                        f"Location: {emotion['location']}"  # Second line
                                    ],
                                    className="card-text center-text",
                                )
                            ])
                        ],
                    ),
                    id=f"{emotion['name']}",
                    style={"width": "18rem", "margin": "auto"},
                    className="clickable-card",
                    n_clicks=0
                ),
                md=4
            ) for emotion in emotions
        ],
        justify="around"
    ),
    style={'margin-top': '20px'}
)

# ========= Learn Tab  Components
learn_card = dbc.Card(
    [
        dbc.CardHeader("Emotional Response Analysis and Reinforcement Learning System Overview"),
        dbc.CardBody(
            html.Div(
                learn_text,
                style={"overflow": "scroll", "height": "700px"}
            )
        )
    ],
    className="mt-4",
)

# ========= Build tabs
tabs = dbc.Tabs(
    [
        dbc.Tab(learn_card, tab_id="tab1", label="Learn"),
        dbc.Tab(
            [asset_allocation_card, time_period_card, input_groups, feedback_leaderboard_table],
            tab_id="tab-2",
            label="Play",
            className="pb-4",
        )
    ],
    id="tabs",
    active_tab="tab-2",
    className="mt-2",
)

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H2(
                    "Embrace AI",
                    className="text-center bg-primary text-white p-2",
                ),
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div([
                #       html.H1("Emotion Cards", className="text-center"),  # Center the title
                    emotion_cards,
                    html.Br(),
                ], className="text-center"),  # Center the content within the div
                #width={"size": 10, "offset": 1}  # Adjust the size and offset to center the column
            ),
            justify="center",  # Ensures the row content is centered
            align="center",  # Vertically centers the row content
            className="h-100"  # Make sure the row takes full height if needed
        ),

        dbc.Row(
            [
                dbc.Col(tabs, width=12, lg=5, className="mt-4 border"),
                dbc.Col(
                    [
                        dcc.Graph(id="live_graph"),
                        dcc.Graph(id='training_time_bar_chart'),
                        html.Div(id="summary_table"),
                    ],
                    width=12,
                    lg=7,
                    className="pt-4",
                ),
            ],
            className="ms-1",
        ),
        dbc.Row(dbc.Col(footer)),
    ],
    fluid=True,
)

"""
==========================================================================
Callbacks
"""
@app.callback(
    [
        Output(f"{emotion['name']}", "style") for emotion in emotions
    ],
    [
        Input(f"{emotion['name']}", "n_clicks") for emotion in emotions
    ],
    prevent_initial_call=True
)
def highlight_active_card(*args):
    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    return [
        {"width": "18rem", "margin": "auto", "filter": "grayscale(100%)"}
        if f"{emotion['name']}" != triggered_id else
        {"width": "18rem", "margin": "auto"}
        for emotion in emotions
    ]

@app.callback(
    Output("start_yr", "value"),
    Output("time_period", "value"),
    Input("start_yr", "value"),
    Input("time_period", "value"),
)
def update_time_period(start_yr, period_number):
    """syncs inputs and selected time periods"""
    ctx = callback_context
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]
    start_yr = 2007
    return start_yr, period_number

@app.callback(
    [Output(f"{emotion['name']}", "children") for emotion in emotions],
    [Input(f"{emotion['name']}", "n_clicks") for emotion in emotions],
    prevent_initial_call=True
)
def update_card_content(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return [no_update] * len(emotions)  # Ensures the list is the same length as the number of emotions
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Using list comprehension to construct updated card contents
    updated_cards = [
        dbc.Card(
            [
                dbc.CardImg(src=app.get_asset_url(emotion['image']), top=True, style={"padding": "10px"}),
                dbc.CardBody([
                    html.Hr(),
                    html.P(
                        "This is a pic of {} emotion.".format(triggered_id.upper()) if emotion['name'].lower() == triggered_id.lower() else "Click on card to classify emotion.",
                        className="card-text center-text"
                    ),
                    html.P(
                        "Location: {}".format(emotion['location']),
                        className="card-text center-text"
                    )
                ])
            ],
        ) for emotion in emotions
    ]
    
    return updated_cards

# Callback to update the graph based on selected radio item
@app.callback(
    Output("live_graph", "figure"),
    Input("time_period", "value")
)
def update_graph(selected_period):
    # Check if the callback is triggered by user interaction or page load
    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] != 'time_period':
        # Return an empty initial graph if not triggered by user interaction
        fig = px.line()
        return fig

    # Get the current feedback score from the selected period's "start_yr"
    feedback_score = mindfulness_feedback_scale[selected_period]["start_yr"]

    # Remove the existing "0 minutes ago" point if it exists
    filtered_data = data.copy()
    filtered_data['Time_Ago'] = [(f"{(datetime.datetime.now() - ts).seconds // 60} minutes ago") for ts in filtered_data['Timestamp']]
    filtered_data = filtered_data[filtered_data['Time_Ago'] != "0 minutes ago"]

    # Add the new point with "0 minutes ago"
    new_point = pd.DataFrame({
        'Timestamp': [datetime.datetime.now()],
        'Feedback_Score': [feedback_score],
        'Emotion': ["New Feedback"],
        'Location': ["User Input"],
        'Time_Ago': ["0 minutes ago"]
    })

    updated_data = pd.concat([filtered_data, new_point], ignore_index=True)

    # Plot with Plotly
    fig = px.line(updated_data, x='Time_Ago', y='Feedback_Score', markers=True, title='TiDB Integrated Live Data Stream Graph - Feedback Scores Over Time',
                  hover_name='Time_Ago',
                  hover_data=['Emotion', 'Location'])
    fig.update_layout(
        xaxis_title='Time (Minutes Ago)',
        yaxis_title='Feedback Score (1-5)',
        xaxis_tickangle=-45,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font_color='#333333',
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='#CCCCCC'),
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='#CCCCCC')
    )
    return fig

def generate_data_training_time_data():
    data = {
        'iteration': list(range(1, 11)),
        'cpu_usage': [random.randint(50, 90) for _ in range(10)],  # CPU usage in percentage
        'ram_usage': [random.randint(16, 32) for _ in range(10)],  # RAM usage in GB
        'fetch_time': [random.randint(10, 50) for _ in range(10)],  # Fetch time in ms
        'data_volume': [random.randint(8000, 12000) for _ in range(10)],  # Data volume in records
        'vector_search_records': [random.randint(5000, 7000) for _ in range(10)]  # Vector search subset
    }
    return data

data_training_time = generate_data_training_time_data()

@app.callback(
    Output("training_time_bar_chart", "figure"),
    Input("time_period", "value")
)
def update_chart_on_feedback(value):
    # Modify the last entry (episode) in the data based on the selected feedback rating
    index = -1  # Index for the last iteration
    data_training_time['cpu_usage'][index] = random.randint(50, 90)
    data_training_time['ram_usage'][index] = random.randint(16, 32)
    data_training_time['fetch_time'][index] = random.randint(10, 50)
    data_training_time['data_volume'][index] = random.randint(8000, 12000)
    data_training_time['vector_search_records'][index] = random.randint(5000, 7000)
    
    fig = go.Figure()

    # Data Volume and Vector Search Records with tooltips including additional info
    fig.add_trace(go.Bar(
        x=data_training_time['iteration'],
        y=data_training_time['data_volume'],
        name='Data Volume (records)',
        marker_color='green',
        hovertemplate=(
            'Iteration: %{x}<br>' +
            'Data Volume: %{y} records<br>' +
            'CPU Usage: %{customdata[0]}%<br>' +
            'RAM Usage: %{customdata[1]} GB<br>' +
            'Fetch Time: %{customdata[2]} ms<br>' +
            '<extra></extra>'
        ),
        customdata=list(zip(data_training_time['cpu_usage'], data_training_time['ram_usage'], data_training_time['fetch_time']))
    ))

    fig.add_trace(go.Bar(
        x=data_training_time['iteration'],
        y=data_training_time['vector_search_records'],
        name='Vector Search Records (records)',
        marker_color='orange',
        hovertemplate=(
            'Iteration: %{x}<br>' +
            'Vector Search Records: %{y} records<br>' +
            'CPU Usage: %{customdata[0]}%<br>' +
            'RAM Usage: %{customdata[1]} GB<br>' +
            'Fetch Time: %{customdata[2]} ms<br>' +
            '<extra></extra>'
        ),
        customdata=list(zip(data_training_time['cpu_usage'], data_training_time['ram_usage'], data_training_time['fetch_time']))
    ))

    fig.update_layout(
        barmode='group',
        title='TiDB Retrieval & Agent Info',
        xaxis_title='Agent Episode Iterations',
        yaxis_title='Metrics',
        legend_title='Metrics',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 11)),  # Explicitly set tick values to show all iterations
            ticktext=[str(i) for i in range(1, 11)]  # Label them with episode numbers 1-10
        )
    )

    return fig

# populate mindfulness exercise
@app.callback(
    Output(asset_allocation_text, "children"),
    [Input(f"{emotion['name']}", "n_clicks") for emotion in emotions],
    prevent_initial_call=True
)
def update_mindfulness_exercise(*args):
    ctx = callback_context
    if not ctx.triggered:
        return no_update

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0].replace('-card', '')

    # Filter leaderboard data based on the emotion clicked
    filtered_exercises = [
        exercise for exercise, emotion in zip(
            leaderboard_data["Exercise"], leaderboard_data["Emotion"]
        ) if emotion.lower() == triggered_id.lower()
    ]

    if filtered_exercises:
        # Randomly pick one exercise from the filtered list
        selected_exercise = random.choice(filtered_exercises)
    else:
        selected_exercise = "No matching exercises found."

    # Update the Markdown with the selected exercise
    return f"""{selected_exercise}"""

# Callback to update the leaderboard based on feedback rating selection
@app.callback(
    Output('leaderboard_table', 'data'),
    Input('time_period', 'value'),
    State('asset_allocation_text', 'children'),
    prevent_initial_call=True
)
def update_leaderboard_table(rating, selected_exercise_text):
    #print("rating", rating)
    global df_leaderboard
    if rating == 0 or selected_exercise_text is None:
        # No update if the rating is 0
        return df_leaderboard.sort_values(by='Q-Value', ascending=False).to_dict('records')
    
    selected_exercise = selected_exercise_text.strip()

    # Get the start_yr value based on the rating index
    start_yr = mindfulness_feedback_scale[rating]["start_yr"]
    exercise_index = df_leaderboard[df_leaderboard['Exercise'] == selected_exercise].index
    print("exercise_index", exercise_index)
    if not exercise_index.empty:
        exercise_index = exercise_index[0]  # Safely get the first index if it exists

        if start_yr == 0:
            # No update if the start_yr is 0
            return df_leaderboard.sort_values(by='Q-Value', ascending=False).to_dict('records')

        # Adjust the Q-value based on the start_yr value
        if start_yr >= 3:
            # Increase Q-value slightly if the rating is 3 or above
            df_leaderboard.at[exercise_index, 'Q-Value'] += 0.02 * (start_yr - 2)
        else:
            # Decrease Q-value slightly if the rating is below 3
            df_leaderboard.at[exercise_index, 'Q-Value'] -= 0.02 * (3 - start_yr)

        # Ensure Q-values stay within valid bounds (0.0 to 1.0)
        df_leaderboard['Q-Value'] = df_leaderboard['Q-Value'].clip(0.0, 1.0)
        # Format the Q-Value column to show values up to the thousandths place
        df_leaderboard['Q-Value'] = df_leaderboard['Q-Value'].round(3)
    
    # Sort by Q-Value in descending order to display the highest Q-value first
    return df_leaderboard.sort_values(by='Q-Value', ascending=False).to_dict('records')

@app.callback(
    Output('response-emotion', 'children'),
    Input('insert-emotion', 'n_clicks'),
    state=[State('input-data', 'value')]
)
def handle_insert_emotion(n_clicks, data):
    if n_clicks is None:
        return ""
    response = insert_emotion_cnn_ai_training({"data": data}, api_key)
    return str(response)

if __name__ == '__main__':
    # Dynamically bind to the port provided by Cloud Run
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
