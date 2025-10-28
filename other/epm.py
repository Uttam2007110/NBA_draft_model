# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 15:52:26 2025
NBA Projections based on data from Dunks and Threes EPM
@author: Subramanya.Ganti
"""
#%% imports
import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup
import re
import json
import ast

pd.set_option('mode.chained_assignment', None)

gw = 2
gd = 0  #set this to 0 to get the whole game week

#%% functions
def get_player_info():
    
    url = "https://nbafantasy.nba.com/api/bootstrap-static/"
    r = requests.get(url,verify=False)
    json = r.json()
    elements = pd.DataFrame(json['elements'])
    elements['name'] = elements['first_name'] + ' ' + elements['second_name']
    teams = pd.DataFrame(json['teams'])
    
    elements = elements[['code','id','name','now_cost','team','element_type']]
    teams = teams[['id','name','short_name']]
    return(elements,teams)

def get_fixture_info(player_info):
    
    fixtures = []
    
    for i in player_info['id']:
        url = "https://nbafantasy.nba.com/api/element-summary/"+str(i)+"/"
        r = requests.get(url,verify=False)
        json = r.json()
        if json == {'detail': 'Not found.'}:
            continue
        else:
            data=pd.DataFrame(json['fixtures'])
            data["id"] = i
            data=data[["team_h", "team_a", "event_name", "is_home", "id"]]
            fixtures.append(data)

    fixtures=pd.concat(fixtures)
    
    return(fixtures)

def extract_epm_data():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
    }
    
    url = 'https://dunksandthrees.com/epm'
    
    try:
        response = requests.get(url, verify=False, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
    
    data = response.text
    soup = BeautifulSoup(data, 'html.parser')
    script_tags = soup.find_all('script')
    script_contents = []
    
    for script in script_tags:
        if script.string:  # Check if the script tag has content
            script_contents.append(script.string.strip())    
    
    pattern = r'\{season:2026,game_dt.*?\}'
    matches = re.findall(pattern, data)
    data_list = [match.strip() for match in matches]
    return data_list

def modify_strings(list_of_s):
    mod_list = []
    for s in list_of_s:
        s = s.replace(',', ',' + '"')
        s = s.replace(':', '"' + ':')
        s = s.replace('{', '{"')
        s = s.replace('null','0')
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
    full['p_mp_48'] *= 1.1
    for t in full['team_alias'].unique():
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

def injury_status():
    injuries = pd.read_html('https://sports.yahoo.com/nba/injuries/')
    injuries = pd.concat(injuries)
    injuries = injuries[['Player','Pos','Status','Date']]
    injuries = injuries.dropna()
    injuries['Player'] = injuries['Player'].str.replace('í','i')
    injuries['Player'] = injuries['Player'].str.replace('č','c')
    injuries['Player'] = injuries['Player'].str.replace('Č','C')
    injuries['Player'] = injuries['Player'].str.replace('ić','ic')
    injuries['Player'] = injuries['Player'].str.replace('ö','o')
    injuries['Player'] = injuries['Player'].str.replace('é','e')
    injuries['Player'] = injuries['Player'].str.replace('ü','u')
    injuries['Player'] = injuries['Player'].str.replace('ņ','n')
    injuries['Player'] = injuries['Player'].str.replace('ģ','g')
    injuries['Player'] = injuries['Player'].str.replace('ô','o')
    injuries['Player'] = injuries['Player'].str.replace('ū','u')
    injuries['Player'] = injuries['Player'].str.replace('Ş','S')
    injuries['Player'] = injuries['Player'].str.replace('Š','S')
    injuries['Player'] = injuries['Player'].str.replace('è','e')
    #injuries['Player'] = injuries['Player'].str.replace('P.J. Washington Jr.','P.J. Washington')
    #injuries['Player'] = injuries['Player'].str.replace('GG Jackson II','GG Jackson')
    #injuries['Player'] = injuries['Player'].str.replace('Xavier Tillman Sr.','Xavier Tillman')
    #injuries['Player'] = injuries['Player'].str.replace('Jeff Dowtin Jr.','Jeff Dowtin')
    #injuries['Player'] = injuries['Player'].str.replace('Craig Porter Jr.','Craig Porter')
    #injuries['Player'] = injuries['Player'].str.replace('Ron Holland II','Ronald Holland II')
    #injuries['Player'] = injuries['Player'].str.replace('Tolu Smith III','Tolu Smith')
    #injuries['Player'] = injuries['Player'].str.replace('Trey Jemison III','Trey Jemison')
    #injuries['Player'] = injuries['Player'].str.replace('AJ Green','A.J. Green')
    #injuries['Player'] = injuries['Player'].str.replace('KJ Simpson','K.J. Simpson')
    #injuries['Player'] = injuries['Player'].str.replace('KJ Martin','Kenyon Martin Jr.')
    return injuries

def matchup_stats(home,away,matchup,team):
    print()
    game_pace = (matchup.loc[matchup['team_alias']==home,'adj_pace'].values[0] + matchup.loc[matchup['team_alias']==away,'adj_pace'].values[0]) / 2
    game_pace *= (98.75/ matchup['adj_pace'].mean()) # 95 for the playoffs
    
    home_pace_factor = game_pace/matchup.loc[matchup['team_alias']==home,'adj_pace'].values[0]
    away_pace_factor = game_pace/matchup.loc[matchup['team_alias']==away,'adj_pace'].values[0]
    
    #team = df_adj.copy()
    team = team[(team['team_alias']==home)|(team['team_alias']==away)]
    team.loc[team['team_alias']==home,'pace factor'] = home_pace_factor
    team.loc[team['team_alias']==away,'pace factor'] = away_pace_factor
    
    home_usage = (team.loc[team['team_alias']==home,'p_usg'] * team.loc[team['team_alias']==home,'p_mp_48']/48).sum()
    team.loc[team['team_alias']==home,'factor'] = (1/home_usage)
    away_usage = (team.loc[team['team_alias']==away,'p_usg'] * team.loc[team['team_alias']==away,'p_mp_48']/48).sum()
    team.loc[team['team_alias']==away,'factor'] = (1/away_usage)
    
    team['pts'] = team['p_pts_100'] * (team['p_t_poss_48']/100) * (team['p_mp_48']/48) * team['factor'] * team['pace factor']
    team['ast'] = team['p_ast_100'] * (team['p_t_poss_48']/100) * (team['p_mp_48']/48) * team['factor'] * team['pace factor']
    team['tov'] = team['p_tov_100'] * (team['p_t_poss_48']/100) * (team['p_mp_48']/48) * team['factor'] * team['pace factor']
    team['orb'] = team['p_orb_100'] * (team['p_t_poss_48']/100) * (team['p_mp_48']/48) * team['factor'] * team['pace factor']
    team['drb'] = team['p_drb_100'] * (team['p_t_poss_48']/100) * (team['p_mp_48']/48) * team['factor'] * team['pace factor']
    team['stl'] = team['p_stl_100'] * (team['p_t_poss_48']/100) * (team['p_mp_48']/48) * team['factor'] * team['pace factor']
    team['blk'] = team['p_blk_100'] * (team['p_t_poss_48']/100) * (team['p_mp_48']/48) * team['factor'] * team['pace factor']
    
    rating_adj = matchup.loc[matchup['team_alias']==home,'rating'].values[0] - matchup.loc[matchup['team_alias']==away,'rating'].values[0] + 2.5
    home_pts = team.loc[team['team_alias']==home,'pts'].sum()
    away_pts = team.loc[team['team_alias']==away,'pts'].sum()
    home_adj = (home_pts + (rating_adj - (home_pts - away_pts))/2)/home_pts
    away_adj = (away_pts - (rating_adj - (home_pts - away_pts))/2)/away_pts
    
    team.loc[team['team_alias']==home,'pts'] *= home_adj
    team.loc[team['team_alias']==home,'ast'] *= home_adj
    team.loc[team['team_alias']==home,'tov'] *= home_adj
    team.loc[team['team_alias']==home,'orb'] *= home_adj
    team.loc[team['team_alias']==home,'drb'] *= home_adj
    team.loc[team['team_alias']==home,'stl'] *= home_adj
    team.loc[team['team_alias']==home,'blk'] *= home_adj
    team.loc[team['team_alias']==home,'opponent'] = away
    team.loc[team['team_alias']==away,'pts'] *= away_adj
    team.loc[team['team_alias']==away,'ast'] *= away_adj
    team.loc[team['team_alias']==away,'tov'] *= away_adj
    team.loc[team['team_alias']==away,'orb'] *= away_adj
    team.loc[team['team_alias']==away,'drb'] *= away_adj
    team.loc[team['team_alias']==away,'stl'] *= away_adj
    team.loc[team['team_alias']==away,'blk'] *= away_adj
    team.loc[team['team_alias']==away,'opponent'] = home
    
    team = team[['player_id', 'player_name', 'team_alias', 'opponent','injury', 'p_mp_48', 'pts', 'ast', 'tov', 'orb', 'drb', 'stl', 'blk']]
    team['EV'] = team['pts'] + team['orb']+ team['drb']+ 2*team['ast']+ 3*team['blk']+ 3*team['stl']
    print(home,round(team.loc[team['team_alias']==home,'pts'].sum(),2))
    print(away,round(team.loc[team['team_alias']==away,'pts'].sum(),2))
    return team

#%% player names
player_names,team_id = get_player_info()
player_names = player_names.loc[player_names['code']>1]
player_names = player_names.sort_values(['now_cost', 'team'], ascending=[False, True])

#%% fixtures by team
#team_list = player_names.groupby('team').first()
#team_list = team_list.reset_index()
fixtures = get_fixture_info(player_names.groupby('team').first().reset_index())
fixtures = fixtures.drop('id',axis=1)
fixtures['event_name'] = fixtures['event_name'].str.replace('Gameweek ', 'GD_')
fixtures['event_name'] = fixtures['event_name'].str.replace(" - Day ", "_") 
fixtures[['event_name', 'gameweek','gameday']] = fixtures['event_name'].str.split('_', expand=True)

gw_fixtures = fixtures[(fixtures['gameweek']==str(gw))&(fixtures['is_home']==True)]
gw_fixtures = gw_fixtures[['team_h', 'team_a', 'gameweek', 'gameday']]
gw_fixtures = gw_fixtures.merge(team_id, left_on=['team_h'], right_on=['id'], how='left')
gw_fixtures = gw_fixtures.merge(team_id, left_on=['team_a'], right_on=['id'], how='left')

if(gd != 0):
    gw_fixtures = gw_fixtures[gw_fixtures['gameday']==str(gd)]
else:
    gw_fixtures = gw_fixtures

#%% extract data from dunks and threes
player_data = extract_epm_data()
player_data = modify_strings(player_data)
player_data = convert_string_list_to_dict(player_data)

injury_report = injury_status()
injury_report[['Status', 'Type']] = injury_report['Status'].str.split('(', expand=True)
injury_report['Type'] = injury_report['Type'].str.replace(")","")
injury_report = injury_report[injury_report['Type']!='Rest']
injury_report['injury'] = 0
injury_report.loc[injury_report['Status']=='Day-To-Day ','injury'] = 0.75

player_data = pd.DataFrame(player_data)
player_data = player_data[['season', 'game_dt', 'player_id', 'player_name', 'team_id',
       'team_alias', 'age', 'inches', 'weight', 'rookie_year', 'position',
       'off', 'def', 'tot', 'p_pct_start', 'p_t_poss_48', 'p_mp_48', 'p_usg',
       'p_pts_100', 'p_tspct', 'p_efg', 'p_fga_rim_100', 'p_fga_mid_100',
       'p_fg2a_100', 'p_fg3a_100', 'p_fta_100', 'p_fgpct_rim', 'p_fgpct_mid',
       'p_fg2pct', 'p_fg3pct', 'p_ftpct', 'p_ast_100', 'p_tov_100',
       'p_orb_100', 'p_drb_100', 'p_stl_100', 'p_blk_100']]

#%% adjust player minutes to 240 per team
df_adj = player_data.copy()
df_adj = df_adj.merge(injury_report[['Player','injury']], left_on='player_name', right_on='Player', how='left')
df_adj['injury'] = df_adj['injury'].fillna(1)
df_adj['p_mp_48'] *= df_adj['injury']
df_adj = mins_adjustment(df_adj)

df_adj['adj_off'] = df_adj['off'] * df_adj['p_mp_48']/48
df_adj['adj_def'] = df_adj['def'] * df_adj['p_mp_48']/48
df_adj['adj_pace'] = df_adj['p_t_poss_48'] * df_adj['p_mp_48']/240

matchup = df_adj.pivot_table(values=['adj_off','adj_def','adj_pace'],index=['team_alias'],aggfunc='sum')
matchup['rating'] = matchup['adj_off'] + matchup['adj_def']
matchup = matchup.reset_index()

#%% game level projections
#game_stats = matchup_stats('OKC', 'SAC', matchup, df_adj.copy())
gw_summary = []
for f in gw_fixtures.values:
    game_stats = matchup_stats(f[6], f[9], matchup, df_adj.copy())
    game_stats['week'] = f[2]
    game_stats['day'] = f[3]
    gw_summary.append(game_stats)
del f

gw_summary = pd.concat(gw_summary)
gw_pivot = pd.pivot_table(gw_summary, index=['player_id','player_name'], columns=['day'], values=['EV'])
gw_pivot.columns = gw_pivot.columns.droplevel(0)
gw_pivot['EV total'] = gw_pivot.sum(axis=1, skipna=True)
gw_pivot = gw_pivot.reset_index()
gw_pivot = gw_pivot.sort_values(by='EV total', ascending=False)
gw_pivot = gw_pivot.merge(player_names[['code','now_cost']], left_on=['player_id'], right_on=['code'], how='left')
gw_pivot = gw_pivot.drop('code', axis=1)
gw_pivot['efficiency'] = gw_pivot['EV total'] / gw_pivot['now_cost'] 

print()
print("the gw_summary and gw_pivot are the dataframes with the results")
print()
print("top 10 players for the chosen time frame are")
print(gw_pivot[['player_name','EV total']].head(10))
print()
print("top 10 cost effective players for the chosen time frame are")
print(gw_pivot.sort_values(by='efficiency', ascending=False).head(10)[['player_name','efficiency']].head(10))
