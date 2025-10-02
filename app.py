import fastf1
import numpy as np
import datetime as dt
import pandas as pd
import country_converter as coco
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, State, Input, Output, no_update, set_props
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from fastf1.core import Laps
import fastf1.plotting
from timple.timedelta import strftimedelta
import flagpy as fp


#latest and upcoming events
fastf1.Cache.enable_cache('.fastf1_cache')
schedule = fastf1.get_event_schedule(2025, include_testing=False, backend='fastf1', force_ergast=False)
eventDates = schedule.loc[:, ['EventDate']]
eventDates['dateTime'] = eventDates.apply(lambda x: pd.to_datetime(eventDates['EventDate'], format='%Y-%M-%d'))
date = np.datetime64('today', 'D')
eventDates.set_index('dateTime', inplace=True)

def findNearest():
    nearest = eventDates.index.get_indexer([date], method='backfill')
    nearestEventDate = eventDates.iloc[nearest]
    scheduleIndexed = schedule.set_index('EventDate') 
    filtered = nearestEventDate.reset_index(drop=True)
    nearestEvent = scheduleIndexed.loc[filtered['EventDate']]
    return nearestEvent

nextCountry = findNearest()['Country'].iloc[0]
nextFlag = fp.get_flag_img(nextCountry)
nextLocation = findNearest()['Location'].iloc[0]
nextEventDate = findNearest().index[0].date()
nextCountryIso = coco.convert(names=nextCountry, to='ISO2')
previousRound = findNearest()['RoundNumber'].iloc[0] - 1
previousEvent = schedule.get_event_by_round(previousRound)
previousCountry = previousEvent['Country']
previousFlag = fp.get_flag_img(previousCountry)
previousLocation = previousEvent['Location']
previousEventDate = previousEvent['EventDate'].date()
previousCountryIso = coco.convert(names=previousCountry, to='ISO2')
session = fastf1.get_session(2025, previousRound, 'Race', backend='fastf1')
session.load()
results = session.results.iloc[0:20].loc[:, ['BroadcastName', 'Position', 'TeamName', 'TeamColor', 'HeadshotUrl']]
position1Name = results['BroadcastName'].iloc[0]
position1Team = results['TeamName'].iloc[0]
position1Photo = results['HeadshotUrl'].iloc[0]
position1Color = results['TeamColor'].iloc[0]
position2Name = results['BroadcastName'].iloc[1]
position2Team = results['TeamName'].iloc[1]
position2Photo = results['HeadshotUrl'].iloc[1]
position2Color = results['TeamColor'].iloc[1]
position3Name = results['BroadcastName'].iloc[2]
position3Team = results['TeamName'].iloc[2]
position3Photo = results['HeadshotUrl'].iloc[2]
position3Color = results['TeamColor'].iloc[2]


#qualifying figure
sessionQ = fastf1.get_session(2025, previousRound, 'Q', backend='fastf1')
sessionQ.load()
drivers = pd.unique(sessionQ.laps['Driver'])
list_fastest_laps = list()
for drv in drivers:
    drvs_fastest_lap = sessionQ.laps.pick_drivers(drv).pick_fastest()
    list_fastest_laps.append(drvs_fastest_lap)
fastest_laps = Laps(list_fastest_laps) \
    .sort_values(by='LapTime') \
    .reset_index(drop=True)
pole_lap = fastest_laps.pick_fastest()
fastest_laps['LapTimeDelta'] = fastest_laps['LapTime'] - pole_lap['LapTime']

team_colors = {
    lap['Team']: fastf1.plotting.get_team_color(lap['Team'], session=sessionQ)
    for _, lap in fastest_laps.iterlaps()
}
figQ = px.bar(
    fastest_laps,
    x="LapTimeDelta",
    y="Driver",
    orientation="h",
    color="Team",
    color_discrete_map=team_colors,
)

figQ.update_yaxes(
    autorange="reversed",
    title_text="Driver",
    dtick=1
)
figQ.update_layout(
    title=dict(
        text="Qualifying Fastest Lap Time Delta",
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=16)
    ),
)


#fastest lap visualization
sessionR = fastf1.get_session(2025, previousRound, 'Race', backend='fastf1')
sessionR.load()
fastest_lap = sessionR.laps.pick_fastest()
car_data = fastest_lap.get_car_data()
car_seconds = car_data['Time'].dt.total_seconds()
car_min = car_seconds.min()
car_max = car_seconds.max()
tickvals = np.arange(np.floor(car_min), np.ceil(car_max)+1, 10)
circuit_info = sessionR.get_circuit_info()
figFastestLap = go.Figure()


figFastestLap.add_trace(go.Scatter(
    x=car_seconds,
    y=car_data['Speed'],
    mode='lines',
    line=dict(color='blue'),
    name=fastest_lap['Driver']
))
figFastestLap.update_layout(
    title=dict(
        text="Fastest Lap of the Race",
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=16)
    ),
    xaxis_title="Time in S",
    yaxis_title="Speed in km/h",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,)
)
def format_time(seconds):
    minutes = int(seconds // 60)
    sec = seconds % 60
    return f"{minutes}:{sec}"

figFastestLap.update_xaxes(
    tickmode="array",
    tickvals=tickvals,
    tickangle=90,
    ticktext=[format_time(t) for t in tickvals],
    title="Lap Time"
)

#team standings visualization

standings = []
short_event_names = []

for _, event in schedule.iterrows():
    event_name, round_number = event["EventName"], event["RoundNumber"]
    short_event_names.append(event_name.replace("Grand Prix", "").strip())

    race = fastf1.get_session(2025, event_name, 'Race', backend='fastf1')
    race.load(laps=False, telemetry=False, weather=False, messages=False)

    sprint = None
    
    if event["EventFormat"] == "sprint_qualifying":
        sprint = fastf1.get_session(2025, event_name, "S")
        sprint.load(laps=False, telemetry=False, weather=False, messages=False)

    for _, driver_row in race.results.iterrows():
        abbreviation, race_points, race_position, team_name = (
            driver_row["Abbreviation"],
            driver_row["Points"],
            driver_row["Position"],
            driver_row["TeamName"],
        )

        sprint_points = 0
        if sprint is not None:
            driver_row = sprint.results[
                sprint.results["Abbreviation"] == abbreviation
            ]
            if not driver_row.empty:
                # We need the values[0] accessor because driver_row is actually
                # returned as a dataframe with a single row
                sprint_points = driver_row["Points"].values[0]

        standings.append(
            {
                "EventName": event_name,
                "RoundNumber": round_number,
                "Driver": abbreviation,
                "Points": race_points + sprint_points,
                "Position": race_position,
                "Team": team_name,
            }
        )
pointsdata = pd.DataFrame(standings)
standings_data = pointsdata.pivot(
    index="Driver", columns="RoundNumber", values="Points"
).fillna(0)

# Sort driver points data
standings_data["total_points"] = standings_data.sum(axis=1)
standings_data = standings_data.sort_values(by="total_points", ascending=True)
total_points = standings_data["total_points"].values
standings_data = standings_data.drop(columns=["total_points"])

position_data = pointsdata.pivot(
    index="Driver", columns="RoundNumber", values="Position"
).fillna("N/A")

team_points = (
    pointsdata.groupby(["Team", "RoundNumber"])["Points"]
    .sum()
    .unstack(fill_value=0)
)

team_points["total_points"] = team_points.sum(axis=1)
team_points = team_points.sort_values(by="total_points", ascending=True)
teamtotal_points = team_points["total_points"].values

figTotal = make_subplots(
    rows=1,
    cols=2,
    column_widths=[0.5, 0.5],
    subplot_titles=("F1 Team Summary", "Total Driver Points"),
)

figTotal.add_trace(
    go.Heatmap(
        x=["Total Points"] * len(team_points),
        y=team_points.index,
        z=teamtotal_points,
        text=teamtotal_points,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorscale="YlGnBu",
        showscale=False,
        zmin=0,
        zmax=teamtotal_points.max(),
    ),
    row=1,
    col=1,
)

figTotal.add_trace(
    go.Heatmap(
        x=["Total Points"] * len(total_points),
        y=standings_data.index,
        z=total_points,
        text=total_points,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorscale="YlGnBu",
        showscale=False,
        zmin=0,
        zmax=total_points.max(),
    ),
    row=1,
    col=2,
)

figTotal.update_yaxes(
    tickmode="array",
    tickvals=team_points.index,
    ticktext=team_points.index,
    row=1,
    col=1
)

figTotal.update_yaxes(
    tickmode="array",
    tickvals=standings_data.index,
    ticktext=standings_data.index,
    row=1,
    col=2
)

# draw track map

lap = session.laps.pick_fastest()
pos = lap.get_pos_data()
circuit_info = session.get_circuit_info()

def rotate(xy, *, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)

track = pos.loc[:, ('X', 'Y')].to_numpy()

track_angle = circuit_info.rotation / 180 * np.pi

rotated_track = rotate(track, angle=track_angle)

figtrack = go.Figure()

figtrack.add_trace(
    go.Scatter(
        x=rotated_track[:, 0],
        y=rotated_track[:, 1],
        mode="lines",
        line=dict(color="blue", width=2),
        name="Track"
    )
)

titletrack=dict(
        text="Track Map",
        x=0.5,
        xanchor='center',
        yanchor='top',)

figtrack.update_layout(
    title=titletrack,
    xaxis=dict(scaleanchor="y", showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    plot_bgcolor="white",
    margin=dict(l=20, r=20, t=40, b=20),
)

offset_vector = [500, 0]

for _, corner in circuit_info.corners.iterrows():
    txt = f"{corner['Number']}{corner['Letter']}"
    offset_angle = corner['Angle'] / 180 * np.pi
    offset_x, offset_y = rotate(offset_vector, angle=offset_angle)
    text_x = corner['X'] + offset_x
    text_y = corner['Y'] + offset_y
    text_x, text_y = rotate([text_x, text_y], angle=track_angle)
    track_x, track_y = rotate([corner['X'], corner['Y']], angle=track_angle)

    figtrack.add_trace(go.Scatter(
        x=[text_x],
        y=[text_y],
        mode="markers+text",
        marker=dict(size=14, color="grey", line=dict(width=1, color="black")),
        text=[txt],
        textposition="middle center",
        textfont=dict(color="white", size=10),
        showlegend=False
    ))

    figtrack.add_trace(go.Scatter(
        x=[track_x, text_x],
        y=[track_y, text_y],
        mode="lines",
        line=dict(color="grey", width=1),
        showlegend=False
    ))

figtrack.update_layout(
    xaxis=dict(scaleanchor="y", showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    plot_bgcolor="white",
    margin=dict(l=20, r=20, t=40, b=20),
)


#layout

header = html.Div([
                html.Img(src='assets/icon.png', style={'width': '400px', 'display': 'block', 'marginLeft': 'auto', 'marginRight': 'auto',}),
                html.H1(
                    "Formula 1 Data Dashboard",
                    style={'textAlign': 'center'}
                ),])

header2 = html.Div([
                html.Img(src=previousFlag, style={'width': '200px', 'display': 'block', 'marginLeft': 'auto', 'marginRight': 'auto', 'marginBottom': '25px', 'marginTop': '10px'}),
                html.H2(
                    f"Latest Race: {previousLocation}, {previousCountry} - {previousEventDate}",
                    style={'textAlign': 'center'}
                ),])

podium = html.Div([
                html.H2(
                    f"Podium - {previousLocation}, {previousCountry}",
                    style={'textAlign': 'center'}
                ),
                html.H4(
                    "Race Results", style={'textAlign': 'center', 'marginBottom': '10px'}
                )])

firstplace = dbc.Card([
        dbc.CardImg(src=position1Photo, top=True),
        dbc.CardBody(
            [
                html.H4('1st Place', className="card-title"),
                html.P(
                    f"{position1Name}\n{position1Team}",
                    className="card-text",
                ),
            ]
        ),
    ],
    style={"width": "90%"}, color='primary', outline=True
)

secondplace = dbc.Card([
        dbc.CardImg(src=position2Photo, top=True),
        dbc.CardBody(
            [
                html.H4('2nd Place', className="card-title"),
                html.P(
                    f"{position2Name}\n{position2Team}",
                    className="card-text",
                ),
            ]
        ),
    ],
    style={"width": "90%"}, color='primary', outline=True
)

thirdplace = dbc.Card([
        dbc.CardImg(src=position3Photo, top=True),
        dbc.CardBody(
            [
                html.H4('3rd Place', className="card-title"),
                html.P(
                    f"{position3Name}\n{position3Team}",
                    className="card-text",
                ),
            ]
        ),
    ],
    style={"width": "90%"}, color='primary', outline=True
)

winners = dbc.Row([
        dbc.Col(podium, lg=12),
    ]), dbc.Row([
        dbc.Col(firstplace, lg=4),
        dbc.Col(secondplace, lg=4),
        dbc.Col(thirdplace, lg=4),])


nextevent = html.Div([
                html.Img(src=nextFlag, style={'width': '200px', 'display': 'block', 'marginLeft': 'auto', 'marginRight': 'auto', 'marginBottom': '25px',}),
                html.H2(
                    f"Next Event - {nextLocation}, {nextCountry}",
                    style={'textAlign': 'center'}
                ),
                html.H4(
                    f"{nextEventDate}", style={'textAlign': 'center'}
                )])

#Run Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server
app.layout = html.Div([
    #row 1
     dbc.Row([
        dbc.Col(header, lg=6, className="p-3 bg-body-secondary rounded-3"),
        dbc.Col(header2, lg=6, className="p-3 bg-body-secondary rounded-3"),
    ], align='center'),
    #row 2
    dbc.Row([
        dbc.Col(winners, lg=4),
        dbc.Col(dcc.Graph(
                id='total-graph',
                figure=figTotal,
                clear_on_unhover=True
            ),
            lg=4
        ),
        dbc.Col(dcc.Graph(
                id='track-graph',
                figure=figtrack,
                clear_on_unhover=True
            ),
            lg=4
        ),
    ], align='center'),
    #row 3
    dbc.Row([
        dbc.Col(dcc.Graph(
                id='qualifying-graph',
                figure=figQ,
                clear_on_unhover=True
            ),
            lg=4
        ),

        dbc.Col(
            dcc.Graph(
                id='fastestlap-graph',
                figure=figFastestLap,
                clear_on_unhover=True
            ),
            lg=4
        ),

        dbc.Col(nextevent,
            lg=4)
    ], align='center')
])


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)