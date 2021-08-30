import pandas as pd
import numpy as np
import lightgbm as lgb
import streamlit as st

st.write("""
# **NFL Game Predictor**

## How to read the results:

### **Margin: Positive numbers are the margins predicted in favor of the Home Team, negative margins are for the Away Team.**


### **Confidence:**

- 1 =  Home Team classified as winner
- 0 =  Pass, too close to classify either as outright winner
- -1 = Away Team classified as winner


#### ** For best results only consider games where the margin and confidence models agree on the winning team.

**


""")

team_dict = {'atl':'Atlanta Falcons', 'buf':'Buffalo Bills', 'car':'Carolina Panthers', 'chi':'Chicago Bears', 'cin':'Cincinnati Bengals', 'cle':'Cleveland Browns', 'clt':'Indianapolis Colts', 'crd':'Arizona Cardinals',
         'dal':'Dallas Cowboys', 'den':'Denver Broncos', 'det':'Detroit Lions', 'gnb':'Green Bay Packers', 'htx':'Houston Texans', 'jax':'Jacksonville Jaguars', 'kan':'Kansas City Chiefs', 'mia':'Miami Dolphins', 
         'min':'Minnesota Vikings', 'nor':'New Orleans Saints', 'nwe':'New England Patriots', 'nyg':'New York Giants', 'nyj':'New York Jets', 'oti':'Tennessee Titans', 'phi':'Philadelphia Eagles',
         'pit':'Pittsburgh Steelers', 'rai':'Las Vegas Raiders', 'ram':'Los Angeles Rams', 'rav':'Baltimore Ravens', 'sdg':'Los Angeles Chargers', 'sea':'Seattle Seahawks', 'sfo':'San Francisco 49ers',
         'tam':'Tampa Bay Buccaneers', 'was':'Washington Football Team'}


df_url = '2020df_week20v2full.csv'
df = pd.read_csv(df_url)
df['Opp_Name'] = df['Opp_Name'].astype('category')
df['Team'] = df['Team'] .astype('category')

#early week/preseason sched
df_url2 = '2021df_pretest.csv'
df2 = pd.read_csv(df_url2)
df2['Opp_Name'] = df2['Opp_Name'].astype('category')
df2['Team'] = df2['Team'] .astype('category')


#new auto date detection v1
today = pd.to_datetime("today")
last_week = today - pd.to_timedelta(2, unit='D')
next_week = today + pd.to_timedelta(3, unit='D')
df2['Date'] = pd.to_datetime(df2['Date'], format='%B %d', errors='coerce') + pd.offsets.DateOffset(years=121)
week = df2.Week.where(((df2['Date'] + pd.to_timedelta(2, unit='D')) < next_week) &
                     (df2['Date'] - pd.to_timedelta(1, unit='D') > last_week)).dropna()
# week = max(week.astype(int))
week = 17

df = df[df["Week"].str.contains("Wild Card|Division|Playoffs|Conf. Champ.|SuperBowl")==False]  
# should be updated to not have to exclude playoff strings


#builds df with last 3 week avgs
df1 = df[pd.to_numeric(df['Week']).between(week - 5, week - 0)]
df1.reset_index(inplace=True)
dfavg = df1.groupby(['Team']).agg([np.average]).copy()
dfavg.columns = ['index', 'Week', 'Result', 'Home', 'Tm',   'Opp',  'OFF1stD',  'OFFTotYd', 'OFFPassY', 'OFFRushY', 'TOOFF',    'DEF1stD',  'DEFTotYd',
                 'DEFPassY',    'DEFRushY', 'TODEF',    'OffenseEP',    'DefenseEP',    'Sp_TmsEP']
dfavg = dfavg.reset_index()
dfavg.drop('index', axis=1, inplace=True)

#builds df with last seasons stats and this seasons pre-season/early week games sched
week2 = 1
preddf = df2[pd.to_numeric(df2['Week']).between(week2, week2)].copy()
preddf.reset_index(inplace=True)
# preddf.columns = df1.columns
preddf.sort_values('Team', inplace=True)
preddf = preddf.drop('Unnamed: 0', axis=1)
replace_cols = ['Tm',   'Opp',  'OFF1stD',  'OFFTotYd', 'OFFPassY', 'OFFRushY', 'TOOFF',    'DEF1stD',  'DEFTotYd', 'DEFPassY', 'DEFRushY', 'TODEF',    'OffenseEP',    'DefenseEP',    'Sp_TmsEP']
preddf[replace_cols] = dfavg[replace_cols].copy()
to_pred = preddf.copy()

# old drops
# X_reg = to_pred.drop(['Date','Week','Result','Tm'], axis=1)

X_reg = to_pred.drop(['Week', 'Home', 'Date', 'Tm', 'Result', 'DEF1stD', 'TOOFF'], axis=1)
y_reg = to_pred['Tm'].copy()
X_class = to_pred.drop(['Week', 'Home', 'Date', 'Tm', 'Result', 'DEFPassY',  'OFFTotYd', 'DEF1stD'], axis=1)
y_class = to_pred['Result'].copy()



regmodel = lgb.Booster(model_file='NFLregmodel_rmse_3_4_.txt')
classmodel = lgb.Booster(model_file='classmodel_new_18_logloss.txt')

reg_preds = regmodel.predict(X_reg)
class_preds = classmodel.predict(X_class)

totals = [to_pred['Team'], to_pred['Opp_Name'], pd.Series(reg_preds), pd.Series(np.round(class_preds))]
df_del = pd.concat(totals, axis=1)
df_del.columns = ['Team','Opp_Name','Pts','W/L']
df_del.Team = df_del.Team.map(team_dict)

pts_dict = dict(zip(df_del.Team,df_del.Pts))
w_l_dict = dict(zip(df_del.Team,df_del['W/L']))
# df_del.Team = df_del.Team.map(team_dict)
opp_pts_dict = dict(zip(df_del.Team,df_del.Pts))
opp_w_l_dict = dict(zip(df_del.Team,df_del['W/L']))

new_dict = dict(zip(to_pred.Team.map(team_dict),to_pred.Opp_Name))


games = pd.DataFrame.from_dict(new_dict, orient='index',
                       columns=['Opps'])
games.index.rename('Team', inplace=True)
games.reset_index(inplace=True)
preddf.reset_index(inplace=True)
to_cat = [games, preddf['Home']]
games_df = pd.concat(to_cat, axis=1)


prepared_df = pd.DataFrame()
prepared_df['Home_Team'] = games_df.Team.where(games_df.Home == 1)
prepared_df['Away_Team'] = games_df.Opps.where(games_df.Home == 1)
prepared_df.dropna(how='all', inplace=True)

prepared_df['Home_Score'] = prepared_df.Home_Team.map(pts_dict)
prepared_df['Away_Score'] = prepared_df.Away_Team.map(opp_pts_dict)
prepared_df['Home_W/L'] = prepared_df.Home_Team.map(w_l_dict)
prepared_df['Away_W/L'] = prepared_df.Away_Team.map(opp_w_l_dict)

prepared_df['Margin'] = prepared_df.Home_Score - prepared_df.Away_Score
prepared_df['Confidence'] = prepared_df['Home_W/L'] - prepared_df['Away_W/L']
prepared_df.Margin = prepared_df.Margin.round(1)

finished_df = prepared_df.drop(['Home_Score', 'Away_Score', 'Home_W/L', 'Away_W/L'], axis=1)
# finished_df.Home_Team = finished_df.Home_Team.map(team_dict)
st.dataframe(finished_df)
# print(reg_preds)