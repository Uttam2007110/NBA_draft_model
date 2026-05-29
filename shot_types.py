# -*- coding: utf-8 -*-
"""
Created on Fri May 29 13:53:22 2026
assited% on shottypes from barttorvik
@author: Subramanya.Ganti
"""

import numpy as np
import pandas as pd
import requests

season = 2026

#path = "C:/Users/uttam/Desktop/Sports/basketball/bart/player"
path = "C:/Users/Subramanya.Ganti/Downloads/Sports/basketball/bart/player"

url = f'https://barttorvik.com/{season}_pbp_playerstat_array.json'
response = requests.get(url, verify=False)
data = response.json()

df = pd.DataFrame(data, columns=['pid','player','team','rimFG','rimFGA','rimasst','midFG','midFGA','midasst','3PFG','3PFGA','3Passt','dunkFG','dunkFGA','dunkasst'])

df['rimasst'] /= df['rimFG']
df['midasst'] /= df['midFG']
df['3Passt'] /= df['3PFG']
df['dunkasst'] /= df['dunkFG']
df['rimasst'] = df['rimasst'].fillna(0)
df['midasst'] = df['midasst'].fillna(0)
df['3Passt'] = df['3Passt'].fillna(0)
df['dunkasst'] = df['dunkasst'].fillna(0)

df = df[['pid','player','team','dunkasst','rimasst','midasst','3Passt']]

df.to_csv(f'{path}/{season}_shot_types.csv')