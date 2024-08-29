import pandas as pd
from flask import Flask, jsonify, render_template, request
import json
import nba_api
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
# import seaborn as sns
import math
import pickle

 
app = Flask(__name__)
@app.route("/")
def index():
    return render_template("nba_app.html", text="")

@app.route("/get_players")
def get_players():
    df = pd.read_csv("2023_nba_player_stats.csv")
    player_names = sorted(df["Player"].tolist())
    return json.dumps(player_names)

 

@app.route('/filter_players')
def filter_players():
    team = request.args.get('team')
    df = pd.read_csv("2023_nba_player_stats.csv")
    df = df[df['Team'] == team]  
    player_names = sorted(df["Player"].tolist())

    return json.dumps(player_names)

@app.route('/get_stats')
def get_stats():
    player = request.args.get('player')
    df = pd.read_csv("2023_nba_player_stats.csv")
    player_stats = df[df['Player'] == player]
    # player_stats = player_stats[['PTS', 'GP', 'FGA', 'Min', 'FG%', 'W']]
    player_stats = player_stats[['W', 'PTS', 'REB', 'AST','STL', 'BLK', 'GP']]

    player_stats['PTS'] = round(player_stats['PTS'] / player_stats['GP'], 1)
    player_stats['REB'] = round(player_stats['REB'] / player_stats['GP'], 1)
    player_stats['AST'] = round(player_stats['AST'] / player_stats['GP'],1)
    player_stats['STL'] = round(player_stats['STL'] / player_stats['GP'],1)
    player_stats['BLK'] = round(player_stats['BLK'] / player_stats['GP'],1)

    player_stats_html = '<table id="table-bordered">'
    header_row = '<tr>'
    for col in player_stats.columns:
        header_row += f'<th>{col}</th>'

    data_rows = ''
    for _, row in player_stats.iterrows():
        data_rows += '<tr>'
        for col in player_stats.columns:
            value = row[col]
            data_rows += f'<td><input type="text" id="{col}" value="{value}" \
                name="{col}" style="width: 30px; height: 30px; text-align: center; border: 1px solid #000;"></td>'      
        data_rows += '</tr>'

    # Combine header and data rows into an HTML table
    player_stats_html = f'<table class="table table-bordered">{header_row}{data_rows}</table>'
    
    # Return the HTML table
    return player_stats_html

 

@app.route('/get_teams')
def get_teams():
    df = pd.read_csv("2023_nba_player_stats.csv")
    teams = np.unique(df['Team']).tolist()
    return jsonify(teams)

@app.route('/get_ranking')
def get_ranking():
    player_name = request.args.get('player')
    # data = pd.read_csv('nba_rankings_2014-2020.csv', encoding='latin-1')
    # data = data[data['SEASON'] == '2019-20']
    # data['THREE'] =  data['3PA']

    PTS = request.args.get('PTS')
    W = request.args.get('W')
    REB = request.args.get('REB')
    AST = request.args.get('AST')
    STL = request.args.get('STL')
    BLK = request.args.get('BLK')
    GP = request.args.get('GP')

    model, scaler = get_model()
    columns = ['W', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'GP']

    # Create an empty DataFrame with the specified columns

    new_player_stats = pd.DataFrame(columns=columns)
    PTS = float(PTS)
    REB = float(REB)
    AST = float(AST)
    STL = float(STL)
    BLK = float(BLK)
    GP = float(GP)

    # Define the new row data
    new_row = {
        'W': str(W),
        'PTS': str(PTS),
        'REB': str(REB),
        'AST': str(AST),
        'STL': str(STL),
        'BLK': str(BLK),
        'GP' : str(GP)
    }

    new_player_stats.loc[0] = new_row
    new_player_stats = scaler.transform(new_player_stats)
    out = model.predict(new_player_stats)[0]
    out = round(out,0)
    return str(out)
 

def get_model():
    data = pd.read_csv('nba_rankings_2014-2020.csv', encoding='latin-1')
    data = data[data['SEASON'] == '2019-20']
    stats = data[['W', 'PTS', 'REB', 'AST','STL', 'BLK', 'GP', 'rankings']]
    stats.dropna()

    Y = stats['rankings']
    X = stats.drop(['rankings'], axis=1)
    scaler = preprocessing.StandardScaler()
    scaled_X = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(scaled_X, Y, test_size=0.5)
    
    clf = LinearRegression()
    clf.fit(x_train, y_train)

    return clf, scaler


if __name__ == '__main__':
    app.run(debug=True)


