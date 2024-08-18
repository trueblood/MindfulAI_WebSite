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
    "cash": "#3cb521",
    "bonds": "#fd7e14",
    "stocks": "#446e9b",
    "inflation": "#cd0200",
    "background": "whitesmoke",
}

"""
==========================================================================
Markdown Text
"""

datasource_text = dcc.Markdown(
    """
    [Data source:](http://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histretSP.html)
    Historical Returns on Stocks, Bonds and Bills from NYU Stern School of
    Business
    """
)
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

# asset_allocation_text = dcc.Markdown(
#     """
# """
# )


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

cagr_text = dcc.Markdown(
    """
    (CAGR) is the compound annual growth rate.  It measures the rate of return for an investment over a period of time, 
    such as 5 or 10 years. The CAGR is also called a "smoothed" rate of return because it measures the growth of
     an investment as if it had grown at a steady rate on an annually compounded basis.
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

"""
==========================================================================
Tables
"""

total_returns_table = dash_table.DataTable(
    id="total_returns",
    columns=[{"id": "Year", "name": "Year", "type": "text"}]
    + [
        {"id": col, "name": col, "type": "numeric", "format": {"specifier": "$,.0f"}}
        for col in ["Cash", "Bonds", "Stocks", "Total"]
    ],
    page_size=15,
    style_table={"overflowX": "scroll"},
)

annual_returns_pct_table = dash_table.DataTable(
    id="annual_returns_pct",
    columns=(
        [{"id": "Year", "name": "Year", "type": "text"}]
        + [
            {"id": col, "name": col, "type": "numeric", "format": {"specifier": ".1%"}}
            for col in df.columns[1:]
        ]
    ),
    data=df.to_dict("records"),
    sort_action="native",
    page_size=15,
    style_table={"overflowX": "scroll"},
)


def make_summary_table(dff):
    """Make html table to show cagr and  best and worst periods"""

    table_class = "h5 text-body text-nowrap"
    cash = html.Span(
        [html.I(className="fa fa-money-bill-alt"), " Cash"], className=table_class
    )
    bonds = html.Span(
        [html.I(className="fa fa-handshake"), " Bonds"], className=table_class
    )
    stocks = html.Span(
        [html.I(className="fa fa-industry"), " Stocks"], className=table_class
    )
    inflation = html.Span(
        [html.I(className="fa fa-ambulance"), " Inflation"], className=table_class
    )

    start_yr = dff["Year"].iat[0]
    end_yr = dff["Year"].iat[-1]

    df_table = pd.DataFrame(
        {
            "": [cash, bonds, stocks, inflation],
            f"Rate of Return (CAGR) from {start_yr} to {end_yr}": [
                cagr(dff["all_cash"]),
                cagr(dff["all_bonds"]),
                cagr(dff["all_stocks"]),
                cagr(dff["inflation_only"]),
            ],
            f"Worst 1 Year Return": [
                worst(dff, "3-mon T.Bill"),
                worst(dff, "10yr T.Bond"),
                worst(dff, "S&P 500"),
                "",
            ],
        }
    )
    return dbc.Table.from_dataframe(df_table, bordered=True, hover=True)


"""
==========================================================================
Figures
"""


def make_pie(slider_input, title):
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Cash", "Bonds", "Stocks"],
                values=slider_input,
                textinfo="label+percent",
                textposition="inside",
                marker={"colors": [COLORS["cash"], COLORS["bonds"], COLORS["stocks"]]},
                sort=False,
                hoverinfo="none",
            )
        ]
    )
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        margin=dict(b=25, t=75, l=35, r=25),
        height=325,
        paper_bgcolor=COLORS["background"],
    )
    return fig


def make_line_chart(dff):
    start = dff.loc[1, "Year"]
    yrs = dff["Year"].size - 1
    dtick = 1 if yrs < 16 else 2 if yrs in range(16, 30) else 5

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["all_cash"],
            name="All Cash",
            marker_color=COLORS["cash"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["all_bonds"],
            name="All Bonds (10yr T.Bonds)",
            marker_color=COLORS["bonds"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["all_stocks"],
            name="All Stocks (S&P500)",
            marker_color=COLORS["stocks"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["Total"],
            name="My Portfolio",
            marker_color="black",
            line=dict(width=6, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dff["Year"],
            y=dff["inflation_only"],
            name="Inflation",
            visible=True,
            marker_color=COLORS["inflation"],
        )
    )
    fig.update_layout(
        title=f"Returns for {yrs} years starting {start}",
        template="none",
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        height=400,
        margin=dict(l=40, r=10, t=60, b=55),
        yaxis=dict(tickprefix="$", fixedrange=True),
        xaxis=dict(title="Year Ended", fixedrange=True, dtick=dtick),
    )
    return fig


"""
==========================================================================
Make Tabs
"""

# Create the card component containing the graph
# model_feedback_performance_card = dbc.Card(
#     [
#         dbc.CardHeader("Real-Time Model Feedback Performance"),
#         dbc.CardBody([
#             dcc.Graph(id='real-time-model-performance-graph'),
#             dcc.Interval(
#                 id='interval-component',
#                 interval=10*1000,  # 10 seconds interval
#                 n_intervals=0
#             )
#         ])
#     ],
#     className="mt-4",
# )


# =======Play tab components

#asset_allocation_card = dbc.Card(asset_allocation_text, className="mt-2")


slider_card = dbc.Card(
    [
        html.H4("First set cash allocation %:", className="card-title"),
        dcc.Slider(
            id="cash",
            marks={i: f"{i}%" for i in range(0, 101, 10)},
            min=0,
            max=100,
            step=5,
            value=10,
            included=False,
        ),
        html.H4(
            "Then set stock allocation % ",
            className="card-title mt-3",
        ),
        html.Div("(The rest will be bonds)", className="card-title"),
        dcc.Slider(
            id="stock_bond",
            marks={i: f"{i}%" for i in range(0, 91, 10)},
            min=0,
            max=90,
            step=5,
            value=50,
            included=False,
        ),
    ],
    body=True,
    className="mt-4",
)

# feedback_leaderboard_table = dbc.Card(
#     [
#         html.H4("Feedback Leaderboard", className="card-title text-center mt-3"),
#         dash_table.DataTable(
#             id='leaderboard_table',
#             columns=[{"name": col, "id": col} for col in df_leaderboard.columns],
#             data=df_leaderboard.to_dict('records'),
#             style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
#             style_cell={'textAlign': 'left'},
#             style_data_conditional=[
#                 {
#                     'if': {'filter_query': '{Source} = "user_feedback_for_that_user"'},
#                     'backgroundColor': 'rgb(220, 220, 220)'
#                 },
#                 {
#                     'if': {'filter_query': '{Source} = "recommended_table"'},
#                     'backgroundColor': 'rgb(245, 245, 245)'
#                 }
#             ],
#             page_size=10
#         ),
#     ],
#     body=True,
#     className="mt-4",
# )

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


time_period_data = [
    {
        "label": f"2007-2008: Great Financial Crisis to {MAX_YR}",
        "start_yr": 2007,
        "planning_time": MAX_YR - START_YR + 1,
    },
    {
        "label": "1999-2010: The decade including 2000 Dotcom Bubble peak",
        "start_yr": 1999,
        "planning_time": 10,
    },
    {
        "label": "1969-1979:  The 1970s Energy Crisis",
        "start_yr": 1970,
        "planning_time": 10,
    },
    {
        "label": "1929-1948:  The 20 years following the start of the Great Depression",
        "start_yr": 1929,
        "planning_time": 20,
    },
    {
        "label": f"{MIN_YR}-{MAX_YR}",
        "start_yr": "1928",
        "planning_time": MAX_YR - MIN_YR + 1,
    },
]


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



# ======= InputGroup components

start_amount = dbc.InputGroup(
    [
        dbc.InputGroupText("Start Amount $"),
        dbc.Input(
            id="starting_amount",
            placeholder="Min $10",
            type="number",
            min=10,
            value=10000,
        ),
    ],
    className="mb-3",
)
start_year = dbc.InputGroup(
    [
        dbc.InputGroupText("Start Year"),
        dbc.Input(
            id="start_yr",
            placeholder=f"min {MIN_YR}   max {MAX_YR}",
            type="number",
            min=MIN_YR,
            max=MAX_YR,
            value=START_YR,
        ),
    ],
    className="mb-3",
)
number_of_years = dbc.InputGroup(
    [
        dbc.InputGroupText("Number of Years:"),
        dbc.Input(
            id="planning_time",
            placeholder="# yrs",
            type="number",
            min=1,
            value=MAX_YR - START_YR + 1,
        ),
    ],
    className="mb-3",
)
end_amount = dbc.InputGroup(
    [
        dbc.InputGroupText("Ending Amount"),
        dbc.Input(id="ending_amount", disabled=True, className="text-black"),
    ],
    className="mb-3",
)
rate_of_return = dbc.InputGroup(
    [
        dbc.InputGroupText(
            "Rate of Return(CAGR)",
            id="tooltip_target",
            className="text-decoration-underline",
        ),
        dbc.Input(id="cagr", disabled=True, className="text-black"),
        dbc.Tooltip(cagr_text, target="tooltip_target"),
    ],
    className="mb-3",
)

input_groups = html.Div(
    [start_year],
    className="mt-4 p-4",
    style={"display": "none"} 
)


# =====  Results Tab components

results_card = dbc.Card(
    [
        dbc.CardHeader("My Portfolio Returns - Rebalanced Annually"),
        html.Div(total_returns_table),
    ],
    className="mt-4",
)

# emotion_cards = html.Div(
#     dbc.Row(
#         [
#             dbc.Col(
#                 dbc.Card(
#                     [
#                         dbc.CardImg(src=app.get_asset_url(emotion['image']), top=True, style={"padding": "10px"}),
#                         dbc.CardBody([
#                             html.Hr(),
#                             html.H4(f"{emotion['name']} Card", className="card-title center-text"),
#                             html.P(f"This is the {emotion['name'].lower()} emotion.", className="card-text center-text"),
#                         ])
#                     ],
#                     id=f"{emotion['name']}",
#                     style={"width": "18rem", "margin": "auto"},
#                     className="clickable-card",
#                     n_clicks=0
#                 ),
#                 md=4
#             ) for emotion in emotions
#         ],
#         justify="around"
#     ),
#     style={'margin-top': '20px'}
# )

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
                                    # style={
                                    #     "font-size": "14px",  # Adjust font size as needed
                                    #     "text-align": "center", 
                                    #     "white-space": "normal",  # Allow wrapping to multiple lines
                                    #     "margin": "0"  # Optional: adjust margins for better spacing
                                    # }
                                ),
                                html.P(
                                    [
                                        f"Location: {emotion['location']}"  # Second line
                                    ],
                                    className="card-text center-text",
                                    # style={
                                    #     "font-size": "14px",  # Adjust font size as needed
                                    #     "text-align": "center", 
                                    #     "white-space": "normal",  # Allow wrapping to multiple lines
                                    #     "margin": "0"  # Optional: adjust margins for better spacing
                                    # }
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




# emotion_cards =html.Div(
#     dbc.Row(
#         [
#             dbc.Col(
#                 dbc.Button(  # Use Button to wrap the Card for click functionality
#                     dbc.Card(
#                         [
#                             dbc.CardImg(src=app.get_asset_url(emotion['image']), top=True, style={"padding": "10px"}),
#                             dbc.CardBody([
#                                 html.Hr(),
#                                 html.H4(f"{emotion['name']} Card", className="card-title center-text"),
#                                 html.P(f"This is the {emotion['name'].lower()} emotion.", className="card-text center-text"),
#                             ])
#                         ],
#                       #  style={"width": "18rem", "margin": "auto"}
#                     ),
#                     id=f"{emotion['name']}",  # Assign ID to Button instead of Card
#                     #style={"padding": "0", "border": "none", "background": "none"},  # Make Button invisible
#                     style={"width": "18rem", "margin": "auto"},
#                     className="clickable-card",
#                     n_clicks=0
#                 ),
#                 md=4
#             ) for emotion in emotions
#         ],
#         justify="around"
#     ),
#     style={'margin-top': '20px'}
# )

data_source_card = dbc.Card(
    [
        dbc.CardHeader("Source Data: Annual Total Returns"),
        html.Div(annual_returns_pct_table),
    ],
    className="mt-4",
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
      #  dbc.Tab([results_card, data_source_card], tab_id="tab-3", label="Results"),
    ],
    id="tabs",
    active_tab="tab-2",
    className="mt-2",
)


"""
==========================================================================
Helper functions to calculate investment results, cagr and worst periods
"""


def backtest(stocks, cash, start_bal, nper, start_yr):
    """calculates the investment returns for user selected asset allocation,
    rebalanced annually and returns a dataframe
    """

    end_yr = start_yr + nper - 1
    cash_allocation = cash / 100
    stocks_allocation = stocks / 100
    bonds_allocation = (100 - stocks - cash) / 100

    # Select time period - since data is for year end, include year prior
    # for start ie year[0]
    dff = df[(df.Year >= start_yr - 1) & (df.Year <= end_yr)].set_index(
        "Year", drop=False
    )
    dff["Year"] = dff["Year"].astype(int)

    # add columns for My Portfolio returns
    dff["Cash"] = cash_allocation * start_bal
    dff["Bonds"] = bonds_allocation * start_bal
    dff["Stocks"] = stocks_allocation * start_bal
    dff["Total"] = start_bal
    dff["Rebalance"] = True

    # calculate My Portfolio returns
    for yr in dff.Year + 1:
        if yr <= end_yr:
            # Rebalance at the beginning of the period by reallocating
            # last period's total ending balance
            if dff.loc[yr, "Rebalance"]:
                dff.loc[yr, "Cash"] = dff.loc[yr - 1, "Total"] * cash_allocation
                dff.loc[yr, "Stocks"] = dff.loc[yr - 1, "Total"] * stocks_allocation
                dff.loc[yr, "Bonds"] = dff.loc[yr - 1, "Total"] * bonds_allocation

            # calculate this period's  returns
            dff.loc[yr, "Cash"] = dff.loc[yr, "Cash"] * (
                1 + dff.loc[yr, "3-mon T.Bill"]
            )
            dff.loc[yr, "Stocks"] = dff.loc[yr, "Stocks"] * (1 + dff.loc[yr, "S&P 500"])
            dff.loc[yr, "Bonds"] = dff.loc[yr, "Bonds"] * (
                1 + dff.loc[yr, "10yr T.Bond"]
            )
            dff.loc[yr, "Total"] = dff.loc[yr, ["Cash", "Bonds", "Stocks"]].sum()

    dff = dff.reset_index(drop=True)
    columns = ["Cash", "Stocks", "Bonds", "Total"]
    dff[columns] = dff[columns].round(0)

    # create columns for when portfolio is all cash, all bonds or  all stocks,
    #   include inflation too
    #
    # create new df that starts in yr 1 rather than yr 0
    dff1 = (dff[(dff.Year >= start_yr) & (dff.Year <= end_yr)]).copy()
    #
    # calculate the returns in new df:
    columns = ["all_cash", "all_bonds", "all_stocks", "inflation_only"]
    annual_returns = ["3-mon T.Bill", "10yr T.Bond", "S&P 500", "Inflation"]
    for col, return_pct in zip(columns, annual_returns):
        dff1[col] = round(start_bal * (1 + (1 + dff1[return_pct]).cumprod() - 1), 0)
    #
    # select columns in the new df to merge with original
    dff1 = dff1[["Year"] + columns]
    dff = dff.merge(dff1, how="left")
    # fill in the starting balance for year[0]
    dff.loc[0, columns] = start_bal
    return dff


def cagr(dff):
    """calculate Compound Annual Growth Rate for a series and returns a formated string"""

    start_bal = dff.iat[0]
    end_bal = dff.iat[-1]
    planning_time = len(dff) - 1
    cagr_result = ((end_bal / start_bal) ** (1 / planning_time)) - 1
    return f"{cagr_result:.1%}"


def worst(dff, asset):
    """calculate worst returns for asset in selected period returns formated string"""

    worst_yr_loss = min(dff[asset])
    worst_yr = dff.loc[dff[asset] == worst_yr_loss, "Year"].iloc[0]
    return f"{worst_yr_loss:.1%} in {worst_yr}"


"""
===========================================================================
Main Layout
"""
# app.layout = dbc.Container(
#     [
#         dbc.Row(
#             dbc.Col(
#                 html.H2(
#                     "Mindful AI",
#                     className="text-center bg-primary text-white p-2",
#                 ),
#             )
#         ),
#     dbc.Row(
#         dbc.Col(
#             html.Div([
#          #       html.H1("Emotion Cards", className="text-center"),  # Center the title
#                 emotion_cards
#             ], className="text-center"),  # Center the content within the div
#             #width={"size": 10, "offset": 1}  # Adjust the size and offset to center the column
#         ),
#         justify="center",  # Ensures the row content is centered
#         align="center",  # Vertically centers the row content
#         className="h-100"  # Make sure the row takes full height if needed
#     ),
#         dbc.Row(
#             [




#                 dbc.Col(tabs, width=12, lg=5, className="mt-4 border"),
#                 dbc.Col(
#                     [
#                         dcc.Graph(id="allocation_pie_chart", className="mb-2"),
#                         dcc.Graph(id="returns_chart", className="pb-4"),
#                         html.Hr(),
#                         html.Div(id="summary_table"),
#                         html.H6(datasource_text, className="my-2"),
#                         dcc.Graph(id="live_graph")
#                     ],
#                     width=12,
#                     className="col-12 pt-4",
#                 ),
#             ],
#             className="ms-1",
#         ),
#         dbc.Row(dbc.Col(footer)),
#     ],
#     fluid=True,
# )
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
                      #  dcc.Graph(id="allocation_pie_chart", className="mb-2"),
                       # dcc.Graph(id="returns_chart", className="pb-4"),
                       # html.Hr(),
                        html.Div(id="summary_table"),
                     #   html.H6(datasource_text, className="my-2")
                     
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

# @app.callback(
#     Output("allocation_pie_chart", "figure"),
#     Input("stock_bond", "value"),
#     Input("cash", "value"),
# )
# def update_pie(stocks, cash):
#     bonds = 100 - stocks - cash
#     slider_input = [cash, bonds, stocks]

#     if stocks >= 70:
#         investment_style = "Aggressive"
#     elif stocks <= 30:
#         investment_style = "Conservative"
#     else:
#         investment_style = "Moderate"
#     figure = make_pie(slider_input, investment_style + " Asset Allocation")
#     return figure


# @app.callback(
#     Output("stock_bond", "max"),
#     Output("stock_bond", "marks"),
#     Output("stock_bond", "value"),
#     Input("cash", "value"),
#     State("stock_bond", "value"),
# )
# def update_stock_slider(cash, initial_stock_value):
#     max_slider = 100 - int(cash)
#     stocks = min(max_slider, initial_stock_value)

#     # formats the slider scale
#     if max_slider > 50:
#         marks_slider = {i: f"{i}%" for i in range(0, max_slider + 1, 10)}
#     elif max_slider <= 15:
#         marks_slider = {i: f"{i}%" for i in range(0, max_slider + 1, 1)}
#     else:
#         marks_slider = {i: f"{i}%" for i in range(0, max_slider + 1, 5)}
#     return max_slider, marks_slider, stocks


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


# @app.callback(
#     Output("total_returns", "data"),
#     Output("returns_chart", "figure"),
#     Output("summary_table", "children"),
#     Output("ending_amount", "value"),
#     Output("cagr", "value"),
#     Input("stock_bond", "value"),
#     Input("cash", "value"),
#     Input("starting_amount", "value"),
#     Input("planning_time", "value"),
#     Input("start_yr", "value"),
# )
# def update_totals(stocks, cash, start_bal, planning_time, start_yr):
#     # set defaults for invalid inputs
#     start_bal = 10 if start_bal is None else start_bal
#     planning_time = 1 if planning_time is None else planning_time
#     start_yr = MIN_YR if start_yr is None else int(start_yr)

#     # calculate valid planning time start yr
#     max_time = MAX_YR + 1 - start_yr
#     planning_time = min(max_time, planning_time)
#     if start_yr + planning_time > MAX_YR:
#         start_yr = min(df.iloc[-planning_time, 0], MAX_YR)  # 0 is Year column

#     # create investment returns dataframe
#     dff = backtest(stocks, cash, start_bal, planning_time, start_yr)

#     # create data for DataTable
#     data = dff.to_dict("records")

#     # create the line chart
#     fig = make_line_chart(dff)

#     summary_table = make_summary_table(dff)

#     # format ending balance
#     ending_amount = f"${dff['Total'].iloc[-1]:0,.0f}"

#     # calcluate cagr
#     ending_cagr = cagr(dff["Total"])

#     return data, fig, summary_table, ending_amount, ending_cagr

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
        #    style={"width": "18rem", "margin": "auto", "padding": "0", "border": "none", "background": "none"}
        ) for emotion in emotions
    ]
    
    return updated_cards

    # return [
    #     dbc.Card(
    #         [
    #             dbc.CardImg(src=app.get_asset_url(emotion['image']), top=True, style={"padding": "10px"}),
    #             dbc.CardBody([
    #                 html.Hr(),
    #                 html.P(f"This is the {emotion['name'].lower()} emotion." if emotion['name'].lower() == button_id else "Click on card to classify emotion.", className="card-text center-text")
    #             ])
    #         ],
    #     ) for emotion in emotions
    # ]

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
# @app.callback(
#     Output("training_time_bar_chart", "figure"),
#     Input("training_time_bar_chart", "id")  # Triggers on app load
# )
# def create_chart(_):
#     data = data_training_time

#     fig = go.Figure()

#     # Data Volume and Vector Search Records with tooltips including additional info
#     fig.add_trace(go.Bar(
#         x=data['iteration'],
#         y=data['data_volume'],
#         name='Data Volume (records)',
#         marker_color='green',
#         hovertemplate=(
#             'Iteration: %{x}<br>' +
#             'Data Volume: %{y} records<br>' +
#             'CPU Usage: %{customdata[0]}%<br>' +
#             'RAM Usage: %{customdata[1]} GB<br>' +
#             'Fetch Time: %{customdata[2]} ms<br>' +
#             '<extra></extra>'
#         ),
#         customdata=list(zip(data['cpu_usage'], data['ram_usage'], data['fetch_time']))
#     ))

#     fig.add_trace(go.Bar(
#         x=data['iteration'],
#         y=data['vector_search_records'],
#         name='Vector Search Records (records)',
#         marker_color='orange',
#         hovertemplate=(
#             'Iteration: %{x}<br>' +
#             'Vector Search Records: %{y} records<br>' +
#             'CPU Usage: %{customdata[0]}%<br>' +
#             'RAM Usage: %{customdata[1]} GB<br>' +
#             'Fetch Time: %{customdata[2]} ms<br>' +
#             '<extra></extra>'
#         ),
#         customdata=list(zip(data['cpu_usage'], data['ram_usage'], data['fetch_time']))
#     ))

#     # Vary fetch time across grouped bars
#     fetch_time_adjusted = [data['fetch_time'][i] + random.randint(-3, 3) for i in range(len(data['fetch_time']))]

#     fig.update_layout(
#         barmode='group',
        
#         title='Q-Learning Data Usage During Decision Processing',
#         xaxis_title='Agent Episode',
#         yaxis_title='Data Records Used',
#         legend_title='Metrics'
#     )

#     return fig

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
#    selected_exercise = asset_allocation_text.children
  #  selected_exercise = asset_allocation_text

  #  print("selected_exercise", selected_exercise)


 #   print("start_yr", start_yr)
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


    # if start_yr == 0:
    #     # No update if the start_yr is 0
    #     return df_leaderboard.sort_values(by='Q-Value', ascending=False).to_dict('records')

    # # Adjust the Q-value based on the start_yr value
    # if start_yr >= 3:
    #     # Increase Q-value slightly if the rating is 3 or above
    #     df_leaderboard.at[exercise_index, 'Q-Value'] += 0.02 * (start_yr - 2)
    # else:
    #     # Decrease Q-value slightly if the rating is below 3
    #     df_leaderboard.at[exercise_index, 'Q-Value'] -= 0.02 * (3 - start_yr)


    # # Adjust the Q-value based on the rating
    # for i in range(len(df_leaderboard)):
    #     if rating >= 3:
    #         # Increase Q-value slightly if the rating is 3 or above
    #         df_leaderboard.at[i, 'Q-Value'] += 0.02
    #     elif rating > 0:
    #         # Decrease Q-value slightly if the rating is below 3
    #         df_leaderboard.at[i, 'Q-Value'] -= 0.02
    
    # # Ensure Q-values stay within valid bounds (0.0 to 1.0)
    # df_leaderboard['Q-Value'] = df_leaderboard['Q-Value'].clip(0.0, 1.0)
    # # Format the Q-Value column to show values up to the thousandths place
    # df_leaderboard['Q-Value'] = df_leaderboard['Q-Value'].round(3)
    
    # Sort by Q-Value in descending order to display the highest Q-value first
    return df_leaderboard.sort_values(by='Q-Value', ascending=False).to_dict('records')

# Remember to mention
"""
Real-Time Data Updates

Type: Live Data Stream Graph, Stream keyword sounds good!!!!
Purpose: Showcase the capability of TiDB to handle real-time data updates, which are crucial for the responsiveness of your RL model.
Data Points: Timestamped entries showing data updates, queries per second, and any relevant metrics that reflect system responsiveness.
"""
# Generate fake data function for demonstration remember to replace with real data
# def generate_fake_data():
#     base_time = datetime.now()
#     data = {
#         "timestamp": [base_time - timedelta(minutes=15 - x) for x in range(15)],
#         "feedback_score": [random.randint(1, 5) if random.random() > 0.2 else 'Did Not Do' for _ in range(15)]
#     }
#     return pd.DataFrame(data)

# # Fetch latest feedback data function remember to replace with real data
# def fetch_latest_feedback_data():
#     return generate_fake_data()

# # Callback to update the graph component
# @app.callback(
#     Output('real-time-model-performance-graph', 'figure'),
#     Input('interval-component', 'n_intervals')
# )
# def update_performance_graph(n):
#     df = fetch_latest_feedback_data()
#     df_filtered = df[df['feedback_score'] != 'Did Not Do']

#     # Create the figure using plotly.graph_objects
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=df_filtered['timestamp'], 
#         y=df_filtered['feedback_score'], 
#         mode='lines+markers',
#         name='Feedback Score'
#     ))

#     fig.update_layout(
#         title='Real-Time Model Feedback Performance',
#         xaxis_title='Time',
#         yaxis_title='Feedback Score',
#         yaxis_range=[0,5]
#     )

#     return fig


# Generate initial static fake data
# base_time = datetime.now()
# data = {
#     "timestamp": [base_time - timedelta(minutes=x) for x in range(15)],
#     "feedback_score": [random.randint(1, 5) for _ in range(15)]
# }
# df = pd.DataFrame(data)

# def relative_time(x):
#     delta = datetime.now() - x
#     if delta.days > 0:
#         return f"{delta.days} days ago"
#     elif delta.seconds > 3600:
#         return f"{delta.seconds // 3600} hours ago"
#     else:
#         return f"{delta.seconds // 60} minutes ago"

# # Apply relative time conversion
# df['relative_time'] = df['timestamp'].apply(relative_time)

# @app.callback(
#     Output('feedback-graph', 'figure'),
#     Input('interval-component', 'n_intervals')
# )
# def update_graph(n):
#     fig = go.Figure()

#     fig.add_trace(go.Scatter(
#         x=df['relative_time'], 
#         y=df['feedback_score'], 
#         mode='lines+markers',
#         name='Feedback Score'
#     ))

#     fig.update_layout(
#         title='Feedback Performance Over Time',
#         xaxis_title='Time Ago',
#         yaxis_title='Feedback Score',
#         yaxis=dict(range=[0, 6]),
#         xaxis=dict(type='category')
#     )

#     return fig










# from dash import Dash, html, dcc
# import plotly.graph_objects as go
# import pandas as pd
# import sqlalchemy
# from dash.dependencies import Input, Output

# # Establish a connection to TiDB
# engine = sqlalchemy.create_engine('mysql+pymysql://user:password@host/dbname')

# app = Dash(__name__)

# def fetch_query_data():
#     query = """
#     SELECT query_time, data_volume, response_time
#     FROM query_log
#     ORDER BY query_time DESC
#     LIMIT 100
#     """
#     df = pd.read_sql(query, con=engine)
#     return df

# @app.callback(
#     Output('query-data-graph', 'figure'),
#     Input('interval-component', 'n_intervals')
# )
# def update_graph(n):
#     df = fetch_query_data()
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=df['query_time'], y=df['data_volume'], name='Data Volume', mode='lines+markers'))
#     fig.add_trace(go.Scatter(x=df['query_time'], y=df['response_time'], name='Response Time', mode='lines+markers'))
#     fig.update_layout(title='Real-Time Query Volume and Response Times', xaxis_title='Time', yaxis_title='Volume/Response Time')
#     return fig

# app.layout = html.Div([
#     dcc.Graph(id='query-data-graph'),
#     dcc.Interval(
#         id='interval-component',
#         interval=30*1000,  # 30 seconds interval
#         n_intervals=0
#     )
# ])




# # Training Time and Database Fetch Time

# from dash import Dash, html, dcc
# import plotly.graph_objects as go
# import pandas as pd
# import sqlalchemy
# from dash.dependencies import Input, Output, State

# # Establish a connection to TiDB
# engine = sqlalchemy.create_engine('mysql+pymysql://user:password@host/dbname')

# app = Dash(__name__)

# def fetch_training_data():
#     query = """
#     SELECT iteration, computation_time, fetch_time, data_volume
#     FROM training_log
#     ORDER BY iteration ASC
#     """
#     df = pd.read_sql(query, con=engine)
#     return df

# @app.callback(
#     Output('training-time-graph', 'figure'),
#     Input('update-button', 'n_clicks')
# )
# def update_training_graph(n_clicks):
#     df = fetch_training_data()
#     fig = go.Figure(data=[
#         go.Bar(name='Computation Time', x=df['iteration'], y=df['computation_time'], hoverinfo='y'),
#         go.Bar(name='Fetch Time', x=df['iteration'], y=df['fetch_time'], hoverinfo='y'),
#         go.Bar(name='Data Volume (Rows)', x=df['iteration'], y=df['data_volume'], hoverinfo='y')
#     ])
#     fig.update_layout(barmode='stack', title='Training, Fetch Times, and Data Volume Per Iteration',
#                       xaxis_title='Iteration', yaxis_title='Time (seconds) / Data Volume (Rows)')
#     return fig

# app.layout = html.Div([
#     html.Button('Update Data', id='update-button', n_clicks=0),
#     dcc.Graph(id='training-time-graph')
# ])

# if __name__ == '__main__':
#     app.run_server(debug=True)


if __name__ == '__main__':
    # Dynamically bind to the port provided by Cloud Run
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
