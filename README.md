# Formula 1 Dashboard

This application provides data visualization using the FastF1 API to showcase the latest Formula 1 Race Results and current team and driver standings.

## Features

Features are integrated into Dash using Plotly for all data visualizations.

## Programming

- Frontend: Plotly Dash, HTML, CSS, Javascript
- Backend: Python, Plotly Dash
- Deployment: Heroku
- Data: Pandas, FastF1 API

## Deployment

The project is already adapted for deployment to Heroku with the necessary requirements and Procfile. Unfortunately, the memory requirements are too large for Heroku's basic dyno services, thus I have removed it from the web. It can easily be run locally by installing per the instructions below.

## Installation

The project can easily be cloned and run locally and/or adapted to a new data set. Contributions and ideas are welcome!

`git clone https://github.com/claythomasfinn/Formula-1-Dashboard/`

## Bugs

The project pulls quite a bit of data from the FastF1 API which takes a little time. Please be patient with loading times. Particularly, because there is not direct API call for team and driver overall standings, the data is compiled from each race event and takes a moment.

## Screenshot

![screenshot](https://github.com/claythomasfinn/Formula-1-Dashboard/blob/main/assets/dashboard-screenshot.png)
