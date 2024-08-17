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
    {"name": "Happy", "image": "images/happy/happy_00b597a317f73e5832275a0a5f9aa8250f1a4450bd55ac20387f2c9d.jpg"},
    {"name": "Sad", "image": "images/sad/sad_01b1763812bc6d9932343b0122aefff73ed1a0cce2f252f8b3a80546.jpg"},
    {"name": "Angry", "image": "images/angry/angry_0b4fb5f008ae748b2bfe8fc4d35af1d37ab56120acbbd135be98fdb5.jpg"}
]

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

asset_allocation_text = dcc.Markdown(
    """
> **Asset allocation** is one of the main factors that drive portfolio risk and returns.   Play with the app and see for yourself!

> Change the allocation to cash, bonds and stocks on the sliders and see how your portfolio performs over time in the graph.
  Try entering different time periods and dollar amounts too.

  Below this add font awesome feedback options and did not attempt button 

  Show excel chart with user feedback and exercise average ratings with past location data.  aka were fetching previous states. we fetch from tidb
"""
)

learn_text = dcc.Markdown(
    """
Using synchronous training for your project, where you're combining a user recommendation table (off-policy) and semantic vector search (on-policy), is a strategic choice, especially when aiming for consistency and stability in your learning updates. Make this talk about the top and bottom graphs and how they relate to tidb serverless and vector and how our RL technology utilizes it










    Past performance certainly does not determine future results, but you can still
    learn a lot by reviewing how various asset classes have performed over time.

    Use the sliders to change the asset allocation (how much you invest in cash vs
    bonds vs stock) and see how this affects your returns.

    Note that the results shown in "My Portfolio" assumes rebalancing was done at
    the beginning of every year.  Also, this information is based on the S&P 500 index
    as a proxy for "stocks", the 10 year US Treasury Bond for "bonds" and the 3 month
    US Treasury Bill for "cash."  Your results of course,  would be different based
    on your actual holdings.

    This is intended to help you determine your investment philosophy and understand
    what sort of risks and returns you might see for each asset category.

    The  data is from [Aswath Damodaran](http://people.stern.nyu.edu/adamodar/New_Home_Page/home.htm)
    who teaches  corporate finance and valuation at the Stern School of Business
    at New York University.

    Check out his excellent on-line course in
    [Investment Philosophies.](http://people.stern.nyu.edu/adamodar/New_Home_Page/webcastinvphil.htm)
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


# def make_summary_table(dff):
#     """Make html table to show cagr and  best and worst periods"""

#     table_class = "h5 text-body text-nowrap"
#     cash = html.Span(
#         [html.I(className="fa fa-money-bill-alt"), " Cash"], className=table_class
#     )
#     bonds = html.Span(
#         [html.I(className="fa fa-handshake"), " Bonds"], className=table_class
#     )
#     stocks = html.Span(
#         [html.I(className="fa fa-industry"), " Stocks"], className=table_class
#     )
#     inflation = html.Span(
#         [html.I(className="fa fa-ambulance"), " Inflation"], className=table_class
#     )

#     start_yr = dff["Year"].iat[0]
#     end_yr = dff["Year"].iat[-1]

#     df_table = pd.DataFrame(
#         {
#             "": [cash, bonds, stocks, inflation],
#             f"Rate of Return (CAGR) from {start_yr} to {end_yr}": [
#                 cagr(dff["all_cash"]),
#                 cagr(dff["all_bonds"]),
#                 cagr(dff["all_stocks"]),
#                 cagr(dff["inflation_only"]),
#             ],
#             f"Worst 1 Year Return": [
#                 worst(dff, "3-mon T.Bill"),
#                 worst(dff, "10yr T.Bond"),
#                 worst(dff, "S&P 500"),
#                 "",
#             ],
#         }
#     )
#     return dbc.Table.from_dataframe(df_table, bordered=True, hover=True)


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

asset_allocation_card = dbc.Card(asset_allocation_text, className="mt-2")

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
        "label": "1: Not Helpful - Exercise felt overwhelming or not useful in managing emotions",
        "start_yr": 1,
        "description": "The exercise did not provide relief or made the situation worse."
    },
    {
        "label": "2: Slightly Helpful - Some minor impact but overall felt irrelevant",
        "start_yr": 2,
        "description": "The exercise had a small positive effect, but was mostly unhelpful."
    },
    {
        "label": "3: Moderately Helpful - Exercise helped manage some emotions but not entirely effective",
        "start_yr": 3,
        "description": "The exercise provided moderate relief but wasn’t fully sufficient."
    },
    {
        "label": "4: Helpful - Exercise significantly improved emotional state",
        "start_yr": 4,
        "description": "The exercise was effective in helping manage emotions with good results."
    },
    {
        "label": "5: Very Helpful - Exercise perfectly suited for the situation and fully relieved stress",
        "start_yr": 5,
        "description": "The exercise was highly effective and perfectly aligned with my needs."
    },
    {
        "label": "Exercise did not apply to my situation - The exercise didn’t fit my current emotional context",
        "start_yr": 0,
        "description": "The exercise wasn’t relevant to what I was experiencing."
    }
]



time_period_card = dbc.Card(
    [
        html.H4(
            "Suggested Mindfulness Exercise Feedback Rating",
            className="card-title",
        ),
        dbc.RadioItems(
            id="time_period",
            options=[
                {"label": period["label"], "value": i}
                for i, period in enumerate(mindfulness_feedback_scale)
            ],
            value=0,
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
                            dbc.CardImg(src=app.get_asset_url(emotion['image']), top=True, style={"padding": "10px"}),
                            dbc.CardBody([
                                html.Hr(),
                                html.H4("Emotion Type?", className="card-title center-text"),
                                html.P("Click on card to classify emotion.", className="card-text center-text")
                            ])
                        ],
                        #style={"width": "18rem", "margin": "auto"}
                    ),
                    id=f"{emotion['name']}",  # Ensure ID is unique and descriptive
                    style={"width": "18rem", "margin": "auto", "padding": "0", "border": "none", "background": "none"},
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
        dbc.CardHeader("An Introduction to Asset Allocation"),
        dbc.CardBody(learn_text),
        dbc.Row(
    dbc.Col(
        html.Div([
            html.H1('Live Data Stream Graph - Feedback Scores Over Time', style={'textAlign': 'center'}),
            html.H2('Rate Your Experience', style={'textAlign': 'center', 'marginBottom': '20px'}),
            html.Div([
                html.Button('1', id='button-1', n_clicks=0, style={'width': '50px', 'height': '50px', 'fontSize': '24px', 'margin': '5px'}),
                html.Button('2', id='button-2', n_clicks=0, style={'width': '50px', 'height': '50px', 'fontSize': '24px', 'margin': '5px'}),
                html.Button('3', id='button-3', n_clicks=0, style={'width': '50px', 'height': '50px', 'fontSize': '24px', 'margin': '5px'}),
                html.Button('4', id='button-4', n_clicks=0, style={'width': '50px', 'height': '50px', 'fontSize': '24px', 'margin': '5px'}),
                html.Button('5', id='button-5', n_clicks=0, style={'width': '50px', 'height': '50px', 'fontSize': '24px', 'margin': '5px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '20px'}),
            html.Div([
                html.Button('Exercise Not Relevant', id='button-not-relevant', n_clicks=0, style={'width': '200px', 'height': '50px', 'fontSize': '24px', 'margin': '5px auto'}),
            ], style={'display': 'flex', 'justifyContent': 'center'}),
        ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px', 'boxShadow': '0 0 10px rgba(0, 0, 0, 0.1)'}),
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
            [asset_allocation_text, time_period_card, input_groups],
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
                    "Mindful AI",
                    className="text-center bg-primary text-white p-2",
                ),
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Div([
                #       html.H1("Emotion Cards", className="text-center"),  # Center the title
                    emotion_cards
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
                        html.Hr(),
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
    ctx = callback_context
    if not ctx.triggered:
        return [no_update for _ in emotions]
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"Triggered ID: {triggered_id}")
    button_id = triggered_id  # Extract emotion name from button ID

    return [
        dbc.Card(
            [
                dbc.CardImg(src=app.get_asset_url(emotion['image']), top=True, style={"padding": "10px"}),
                dbc.CardBody([
                    html.Hr(),
                   # html.H4(f"{emotion['name']} Card" if emotion['name'].lower() == button_id else "Emotion Type?", className="card-title center-text"),
                    html.P(f"This is the {emotion['name'].lower()} emotion." if emotion['name'].lower() == button_id else "Click on card to classify emotion.", className="card-text center-text")
                ])
            ],
           # style={"width": "18rem", "margin": "auto"}
        ) for emotion in emotions
    ]

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
    Input("training_time_bar_chart", "id")  # Triggers on app load
)
def create_chart(_):
    data = data_training_time

    fig = go.Figure()

    # Data Volume and Vector Search Records with tooltips including additional info
    fig.add_trace(go.Bar(
        x=data['iteration'],
        y=data['data_volume'],
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
        customdata=list(zip(data['cpu_usage'], data['ram_usage'], data['fetch_time']))
    ))

    fig.add_trace(go.Bar(
        x=data['iteration'],
        y=data['vector_search_records'],
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
        customdata=list(zip(data['cpu_usage'], data['ram_usage'], data['fetch_time']))
    ))

    # Vary fetch time across grouped bars
    fetch_time_adjusted = [data['fetch_time'][i] + random.randint(-3, 3) for i in range(len(data['fetch_time']))]

    fig.update_layout(
        barmode='group',
        
        title='Q-Learning Data Usage During Decision Processing',
        xaxis_title='Agent Episode',
        yaxis_title='Data Records Used',
        legend_title='Metrics'
    )

    return fig


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


if __name__ == "__main__":
    app.run_server(debug=True, port=8051)
