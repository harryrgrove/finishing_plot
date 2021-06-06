import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from progress.bar import ChargingBar
from datetime import datetime, timedelta


def plot(collect_new_data=True, source='StatsBomb', league='PL',
         window=50, player_colors=None, print_names=False, axis='volume', annotate=True):
    """
    :param collect_new_data:
        If True, refreshes data for data.json
    :param source:
        StatsBomb or Understat data
    :param league:
        PL, Bundesliga, Ligue_1, La_Liga, Serie_A data
    :param window:
        Rolling data window size
    :param print_names:
        Print names of players (useful to check spelling)
    :param player_colors:
        Dict of highlighted data
    :param axis:
        Plot by 'volume' or 'time' (time for StatsBomb data only)
    :param annotate:
        Label player names for coloured lines
    """

    if collect_new_data:
        if source == 'StatsBomb':
            df = pd.read_csv(f'SB_data/{league}_data.csv').sort_values(by='date')
            players = df['player'].unique()

            volume_lines = {}
            time_lines = {}
            bar = ChargingBar('Loading Data', max=len(players))
            for player in players:
                bar.next()
                if print_names:
                    print(player)
                player_df = pd.DataFrame(df[df['player'] == player]) # Filter data per player
                player_df['shots'] -= player_df['pk_attempts'] # Exclude penalty data
                player_df['goals'] -= player_df['pk_scored']

                # Build 50-shot rolling averages
                if player_df['shots'].sum() >= window:
                    dates, match_goal_counts, match_xg_counts, shot_goal_counts, shot_xg_counts = [], [], [], [], []
                    run_start, run_end = 0, 0
                    shot_list = []
                    for iloc, i in enumerate(player_df.index):
                        if sum(shot_list) < window:
                            shot_list.append(player_df.loc[i, 'shots'])
                        else:
                            run_end = iloc
                            break
                    for i in range(run_end, len(player_df)):
                        dates.append(player_df.iloc[i]['date'])
                        for r in range(shot_list[-1]):
                            shot_goal_counts.append(sum(player_df['goals'][run_start:run_end]))
                            shot_xg_counts.append(sum(player_df['npxg'][run_start:run_end]))
                        match_goal_counts.append(sum(player_df['goals'][run_start:run_end]))
                        match_xg_counts.append(sum(player_df['npxg'][run_start:run_end]))
                        last_removed = 0
                        shot_list.append(player_df.iloc[i]['shots'])
                        run_end += 1
                        while sum(shot_list) >= window:
                            last_removed = shot_list[0]
                            shot_list = shot_list[1:]
                            run_start += 1
                        shot_list.insert(0, last_removed)
                        run_start -= 1

                    volume_lines[player_df['player'].iloc[0]] = list((np.array(shot_goal_counts) -
                                                                   np.array(shot_xg_counts)) / window)
                    time_lines[player_df['player'].iloc[0]] = {d: (match_goal_counts[i] - match_xg_counts[i]) / window
                                                               for i, d in enumerate(dates)}
            bar.finish()

            with open('volume_data.json', 'w') as outfile:
                json.dump(volume_lines, outfile)
            with open('time_data.json', 'w') as outfile:
                json.dump(time_lines, outfile)

        elif source == 'Understat':
            seasons = np.arange(2014, 2021)
            dfs = []
            for season in seasons:
                path = f'US_data/{league}/{season}'
                for sub_dir, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('shot_df.pickle'):
                            df = pd.read_pickle(os.path.join(sub_dir, file))
                            dfs.append(df)
            df = pd.concat(dfs).sort_values(by='date')
            players = df['name'].unique().tolist()

            lines = {}
            bar = ChargingBar('Loading Data', max=len(players))
            for player in players:
                bar.next()
                player_df = pd.DataFrame(df[(df['name'] == player) & (df['situation'] != 'Penalty')])

                # Build 50-shot rolling averages
                if (player_df['season'] >= 2017).sum() >= window:
                    player_df['+/-'] = (player_df['result'] == 'Goal') - player_df['xG']
                    player_df['rolling'] = player_df['+/-'].rolling(window).mean()
                    player_df = player_df[2017 <= player_df['season']].sort_values(by='date').iloc[window:]
                    lines[player] = player_df['rolling'].tolist()
            bar.finish()

            with open('volume_data.json', 'w') as outfile:
                json.dump(lines, outfile)
        else:
            raise Exception('Invalid Data Source')

    if axis == 'volume':
        with open('volume_data.json') as json_file:
            lines = json.load(json_file)

        # Draw lines
        bar = ChargingBar('Drawing Lines', max=len(lines))
        for player in lines:
            bar.next()
            sns.lineplot(x=range(len(lines[player])), y=lines[player], color='#c0c0c0', alpha=0.5)
        bar.finish()

        plt.axhline(0, ls='dashed', color='k', linewidth=1)

        if player_colors is None:
            player_colors = {
                'Sadio Mané': '#da2b35',
                'Jamie Vardy': 'darkblue'
            }

        for player in player_colors:
            try:
                sns.lineplot(x=range(len(lines[player])), y=lines[player], color=player_colors[player])
                if annotate:
                    plt.annotate(player, xy=(len(lines[player]), lines[player][-1] - 0.01), color=player_colors[player],
                                 size=5, va="top", ha="left", bbox=dict(boxstyle="round", fc="w",
                                                                        edgecolor=player_colors[player]))
            except KeyError:
                pass

        # Set graphical params
        plt.ylim(-0.21, 0.31)
        plt.ylabel(f'Average of previous {window} shots', style='italic')
        plt.xlabel('Shots (non-penalty)', style='italic')
        plt.title(f'Over/Under Goals to xG ({league} since 2017-18)', {'fontname': 'Helvetica'}, size=12)
        plt.text(0.99, 0.01, f'Data via {source}\n@ff_trout',
                 fontsize=8, color='gray',
                 ha='right', va='bottom', alpha=0.5, transform=plt.gca().transAxes)
        plt.show()

    elif axis == 'time':
        if source == 'Understat':
            raise Exception('time axis only available for Statsbomb data')

        with open('time_data.json') as json_file:
            lines = json.load(json_file)

        # Draw lines
        bar = ChargingBar('Drawing Lines', max=len(lines))
        for player in lines:
            bar.next()
            sns.lineplot(x=[datetime.strptime(k, '%Y-%m-%d') for k in lines[player].keys()], y=lines[player].values(),
                         color='#c0c0c0', alpha=0.5)
        bar.finish()

        plt.axhline(0, ls='dashed', color='k', linewidth=1)

        if player_colors is None:
            player_colors = {
                'Sadio Mané': '#da2b35',
                'Jamie Vardy': 'darkblue'
            }

        dates = []

        for player in player_colors:
            try:
                sns.lineplot(x=[datetime.strptime(date, '%Y-%m-%d') for date in lines[player].keys()],
                             y=lines[player].values(), color=player_colors[player])
                if annotate:
                    plt.annotate(player, xy=(datetime.strptime(list(lines[player].keys())[-1], '%Y-%m-%d'),
                                             list(lines[player].values())[-1] - 0.01),
                                 color=player_colors[player], size=5, va="bottom", ha="left",
                                 bbox=dict(boxstyle="round", fc="w", edgecolor=player_colors[player]))
                    for date in list(lines[player].keys()):
                        dates.append(datetime.strptime(date, '%Y-%m-%d'))
            except KeyError:
                pass

        # Set graphical params
        try:
            plt.xlim(min(dates) - timedelta(days=50), max(dates))
        except ValueError:
            pass
        plt.ylim(-0.21, 0.31)
        plt.ylabel('Average of previous 50 shots', style='italic')
        plt.xlabel('Shots (non-penalty)', style='italic')
        plt.title(f'Over/Under Goals to xG ({league} since 2017-18)', {'fontname': 'Helvetica'}, size=12)
        plt.text(0.99, 0.01, f'Data via {source}\n@ff_trout',
                 fontsize=8, color='gray',
                 ha='right', va='bottom', alpha=0.5, transform=plt.gca().transAxes)
        plt.show()
    else:
        raise Exception('Invalid Axis')




plot(
    collect_new_data=True,
    source='StatsBomb',
    league='PL',
    window=50,
    player_colors={
        'Sadio Mané': '#da2b35',
        'Jamie Vardy': 'darkblue'
    },
    axis='volume',
    annotate=True
)
