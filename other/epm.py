# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 15:52:26 2025

@author: Subramanya.Ganti
"""
import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup
import re
import json
import ast

def extract_epm_data():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
    }
    
    # The URL for the Estimated Plus-Minus (EPM) leaderboard
    url = 'https://dunksandthrees.com/epm?m=p_mp_48&team=CHA'
    
    # Send a GET request to the URL
    try:
        response = requests.get(url, verify=False, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
    
    # Parse the HTML content of the page with BeautifulSoup
    data = response.text
    soup = BeautifulSoup(data, 'html.parser')
    
    script_tags = soup.find_all('script')
    script_contents = []
    
    for script in script_tags:
        if script.string:  # Check if the script tag has content
            script_contents.append(script.string.strip())    
    
    # Find all occurrences after 'type:"data"'
    pattern = r'\{season:2026,game_dt.*?\}'
    matches = re.findall(pattern, data)
    
    # Store in a list
    data_list = [match.strip() for match in matches]
    return data_list

def modify_strings(list_of_s):
    mod_list = []
    for s in list_of_s:
        s = s.replace(',', ',' + '"')
        s = s.replace(':', '"' + ':')
        s = s.replace('{', '{"')
        mod_list.append(s)
    return mod_list


def convert_string_list_to_dict(string_list):
    """
    Converts a list of strings, each representing a dictionary-like structure,
    into a list of actual Python dictionaries.
    """
    result_dicts = []
    for s in string_list:
        try:
            # Safely evaluate the string as a Python literal (dictionary)
            evaluated_dict = ast.literal_eval(s)
            if isinstance(evaluated_dict, dict):
                result_dicts.append(evaluated_dict)
            else:
                print(f"Warning: '{s}' did not evaluate to a dictionary.")
        except (ValueError, SyntaxError) as e:
            print(f"Error evaluating string '{s}': {e}")
    return result_dicts

def mins_adjustment(full):
    adjusted = []
    full['p_mp_48'] *= 1.2
    for t in df['team_alias'].unique():
        print(t)
        outfielders = full[full['team_alias']==t]
        outfielders = outfielders.sort_values(by='p_mp_48', ascending=False)
        outfielders['rank'] = list(range(1,len(outfielders)+1))
        outfielders.loc[outfielders['rank']>13,'p_mp_48'] = 0
        exp = 1.0
        while((outfielders['p_mp_48'].sum() <= 235) or (outfielders['p_mp_48'].sum() >= 245)):
            #print(outfielders['p_mp_48'].sum())
            if(outfielders['p_mp_48'].sum() <= 235):
                outfielders['p_mp_48'] *= pow(exp,outfielders['rank']/2)
                exp += 0.001
            elif(outfielders['p_mp_48'].sum() >= 245):
                outfielders['p_mp_48'] *= pow(exp,outfielders['rank']/2)
                exp -= 0.001
            outfielders['p_mp_48'] = outfielders['p_mp_48'].clip(upper=48)
        outfielders['p_mp_48'] = 240*outfielders['p_mp_48']/outfielders['p_mp_48'].sum()
        adjusted.append(outfielders)
    adjusted = pd.concat(adjusted)
    return adjusted

player_data = extract_epm_data()
player_data = modify_strings(player_data)
player_data = convert_string_list_to_dict(player_data)

df = pd.DataFrame(player_data)
df = df[['season', 'game_dt', 'player_id', 'player_name', 'team_id',
       'team_alias', 'age', 'inches', 'weight', 'rookie_year', 'position',
       'off', 'def', 'tot', 'p_pct_start', 'p_t_poss_48', 'p_mp_48', 'p_usg',
       'p_pts_100', 'p_tspct', 'p_efg', 'p_fga_rim_100', 'p_fga_mid_100',
       'p_fg2a_100', 'p_fg3a_100', 'p_fta_100', 'p_fgpct_rim', 'p_fgpct_mid',
       'p_fg2pct', 'p_fg3pct', 'p_ftpct', 'p_ast_100', 'p_tov_100',
       'p_orb_100', 'p_drb_100', 'p_stl_100', 'p_blk_100']]

df_adj = mins_adjustment(df.copy())
df_adj['adj_off'] = df_adj['off'] * df_adj['p_mp_48']/48
df_adj['adj_def'] = df_adj['def'] * df_adj['p_mp_48']/48
df_adj['adj_pace'] = df_adj['p_t_poss_48'] * df_adj['p_mp_48']/240

matchup = df_adj.pivot_table(values=['adj_off','adj_def','adj_pace'],index=['team_alias'],aggfunc='sum')
matchup['rating'] = matchup['adj_off'] + matchup['adj_def']
matchup = matchup.reset_index()

home = 'GSW'
away = 'LAL'

game_pace = (matchup.loc[matchup['team_alias']==home,'adj_pace'].values[0] * matchup.loc[matchup['team_alias']==away,'adj_pace'].values[0]) / matchup['adj_pace'].mean()
bbref = pd.read_html('https://www.basketball-reference.com/leagues/NBA_stats_per_game.html')[0]
bbref.columns = bbref.columns.droplevel(0)
bbref = bbref[['Season','G','Pace']]
bbref = bbref.dropna()
bbref = bbref[bbref['Season'] != 'Season']
bbref = bbref.apply(pd.to_numeric, errors='ignore')
bbref['weight'] = np.exp(-bbref.index)
bbref['weight'] *= bbref['G']

team = df_adj.copy()
team = team[team['team_alias']=='GSW']
team['p_mp_48'] *= 240/team['p_mp_48'].sum()
usage = (team['p_usg'] * team['p_mp_48']/48).sum()
team['factor'] = (1/usage)
team['pts'] = team['p_pts_100'] * (team['p_t_poss_48']/100) * (team['p_mp_48']/48) * team['factor']
team['ast'] = team['p_ast_100'] * (team['p_t_poss_48']/100) * (team['p_mp_48']/48) * team['factor']
team['tov'] = team['p_tov_100'] * (team['p_t_poss_48']/100) * (team['p_mp_48']/48) * team['factor']
team['orb'] = team['p_orb_100'] * (team['p_t_poss_48']/100) * (team['p_mp_48']/48) * team['factor']
team['drb'] = team['p_drb_100'] * (team['p_t_poss_48']/100) * (team['p_mp_48']/48) * team['factor']
team['stl'] = team['p_stl_100'] * (team['p_t_poss_48']/100) * (team['p_mp_48']/48) * team['factor']
team['blk'] = team['p_blk_100'] * (team['p_t_poss_48']/100) * (team['p_mp_48']/48) * team['factor']
team = team[['player_id', 'player_name', 'team_alias', 'p_mp_48', 'pts', 'ast', 'tov', 'orb', 'drb', 'stl', 'blk']]