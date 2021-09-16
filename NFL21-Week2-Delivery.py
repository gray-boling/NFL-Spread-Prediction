import pandas as pd
import numpy as np
import lightgbm as lgb
import streamlit as st

st.title('NFL Spread Predictor')

st.caption("""
# **How to read the results:**

## **Margin:**

### **Positive numbers are margins predicted in favor of the Home Team. Negative margins are predictions for the Away Team.**


## **Confidence:**

### **- 1 =  Home Team classified as winner**
### **- 0 =  Pass, too close to classify either as outright winner**
### **- -1 = Away Team classified as winner**

""")


st.write("""
### **For best results only consider games where the margin and confidence models agree on the winning team.**
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
df_url2 = 'NFL21_week1_full.csv'
df2 = pd.read_csv(df_url2)
df2['Opp_Name'] = df2['Opp_Name'].astype('category')
df2['Team'] = df2['Team'] .astype('category')


#new auto date detection v1
today = pd.to_datetime("today")
last_week = today - pd.to_timedelta(6, unit='D')
next_week = today + pd.to_timedelta(6, unit='D')
df2['Date'] = pd.to_datetime(df2['Date'], format='%B %d', errors='coerce') + pd.offsets.DateOffset(years=121)
week2 = df2.Week.where((df2['Date'] + pd.to_timedelta(4, unit='D') < next_week) & 
                      (df2['Date'] - pd.to_timedelta(5, unit='D') > last_week)).dropna()
week2 = max(week2.astype(int))
week = 17 #used to avg stats from last year for early week game predictions

df = df[df["Week"].str.contains("Wild Card|Division|Playoffs|Conf. Champ.|SuperBowl")==False]  
# should be updated to not have to exclude playoff strings

#builds df with last X week avgs
df1 = df[pd.to_numeric(df['Week']).between(week - 4, week - 0)]
df1.reset_index(inplace=True)

#2021 stats for avg, use first weeks only then comment out
dfearly = df2.where(df2.Week == week2 - 1)
dfearly.reset_index(inplace=True)
avgs = [df1, dfearly]
df4 = pd.concat(avgs)

#change to df1 after first few weeks #builds avg df with last season and early weeks stats
dfavg = df4.groupby(['Team']).agg([np.average]).copy()
dfavg.columns = ['index', 'Week', 'Result', 'Home', 'Tm',   'Opp',  'OFF1stD',  'OFFTotYd', 'OFFPassY', 'OFFRushY', 'TOOFF',    'DEF1stD',  'DEFTotYd',
                 'DEFPassY',    'DEFRushY', 'TODEF',    'OffenseEP',    'DefenseEP',    'Sp_TmsEP']
dfavg = dfavg.reset_index()
dfavg.drop('index', axis=1, inplace=True)

#builds df with last seasons stats and this seasons pre-season/early week games sched
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

X_reg = to_pred.drop(['index', 'Week', 'Home', 'Date', 'Tm', 'Result', 'DEF1stD', 'TOOFF'], axis=1)
y_reg = to_pred['Tm'].copy()
X_class = to_pred.drop(['index', 'Week', 'Home', 'Date', 'Tm', 'Result', 'DEFPassY',  'OFFTotYd', 'DEF1stD'], axis=1)
y_class = to_pred['Result'].copy()


#loading models
regmodel = lgb.Booster(model_file='NFLregmodel_rmse_3_27_home.txt')
classmodel = lgb.Booster(model_file='classmodel-17-5-home.txt')

#inference
reg_preds = regmodel.predict(X_reg)
class_preds = classmodel.predict(X_class)

#building df with predictions for user readability
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
# st.dataframe(finished_df)


#building df with team logos

#@title Attach images to dataframe by team/ test
# from IPython.display import HTML

#Link to .csv file with links to logos per team
logos = pd.read_csv("https://raw.githubusercontent.com/leesharpe/nfldata/master/data/logos.csv")

#dict to link each team's name to the correct url
logos_dict = {'Arizona Cardinals': 'https://upload.wikimedia.org/wikipedia/en/thumb/7/72/Arizona_Cardinals_logo.svg/179px-Arizona_Cardinals_logo.svg.png', 
              'Buffalo Bills': 'https://upload.wikimedia.org/wikipedia/en/thumb/7/77/Buffalo_Bills_logo.svg/189px-Buffalo_Bills_logo.svg.png', 
              'Atlanta Falcons': 'https://upload.wikimedia.org/wikipedia/en/thumb/c/c5/Atlanta_Falcons_logo.svg/192px-Atlanta_Falcons_logo.svg.png', 
              'Baltimore Ravens': 'https://upload.wikimedia.org/wikipedia/en/thumb/1/16/Baltimore_Ravens_logo.svg/193px-Baltimore_Ravens_logo.svg.png', 
              'Cincinnati Bengals':  'https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Cincinnati_Bengals_logo.svg/100px-Cincinnati_Bengals_logo.svg.png',
              'Carolina Panthers': 'https://upload.wikimedia.org/wikipedia/en/thumb/1/1c/Carolina_Panthers_logo.svg/100px-Carolina_Panthers_logo.svg.png',
              'Cleveland Browns':  'https://upload.wikimedia.org/wikipedia/en/thumb/d/d9/Cleveland_Browns_logo.svg/100px-Cleveland_Browns_logo.svg.png',
              'Indianapolis Colts': 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Indianapolis_Colts_logo.svg/100px-Indianapolis_Colts_logo.svg.png', 
              'Chicago Bears': 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Chicago_Bears_logo.svg/100px-Chicago_Bears_logo.svg.png', 
              'Dallas Cowboys':  'https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Dallas_Cowboys.svg/100px-Dallas_Cowboys.svg.png',
              'Denver Broncos':  'https://upload.wikimedia.org/wikipedia/en/thumb/4/44/Denver_Broncos_logo.svg/100px-Denver_Broncos_logo.svg.png',
              'Detroit Lions':  'https://upload.wikimedia.org/wikipedia/en/thumb/7/71/Detroit_Lions_logo.svg/100px-Detroit_Lions_logo.svg.png',
              'Green Bay Packers': 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Green_Bay_Packers_logo.svg/100px-Green_Bay_Packers_logo.svg.png',
              'Houston Texans': 'https://upload.wikimedia.org/wikipedia/en/thumb/2/28/Houston_Texans_logo.svg/100px-Houston_Texans_logo.svg.png',
              'Jacksonville Jaguars': 'https://upload.wikimedia.org/wikipedia/en/thumb/7/74/Jacksonville_Jaguars_logo.svg/100px-Jacksonville_Jaguars_logo.svg.png', 
              'Kansas City Chiefs': 'https://upload.wikimedia.org/wikipedia/en/thumb/e/e1/Kansas_City_Chiefs_logo.svg/100px-Kansas_City_Chiefs_logo.svg.png',
              'Miami Dolphins': 'https://upload.wikimedia.org/wikipedia/en/thumb/3/37/Miami_Dolphins_logo.svg/100px-Miami_Dolphins_logo.svg.png',
              'Minnesota Vikings': 'https://upload.wikimedia.org/wikipedia/en/thumb/4/48/Minnesota_Vikings_logo.svg/98px-Minnesota_Vikings_logo.svg.png', 
              'New Orleans Saints': 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/New_Orleans_Saints_logo.svg/98px-New_Orleans_Saints_logo.svg.png', 
              'New England Patriots': 'https://upload.wikimedia.org/wikipedia/en/thumb/b/b9/New_England_Patriots_logo.svg/100px-New_England_Patriots_logo.svg.png', 
              'New York Giants': 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/New_York_Giants_logo.svg/100px-New_York_Giants_logo.svg.png', 
              'New York Jets': 'https://upload.wikimedia.org/wikipedia/en/thumb/6/6b/New_York_Jets_logo.svg/100px-New_York_Jets_logo.svg.png', 
              'Tennessee Titans': 'http://www.nflgamedata.com/ten.png', 
              'Philadelphia Eagles': 'https://upload.wikimedia.org/wikipedia/en/thumb/8/8e/Philadelphia_Eagles_logo.svg/100px-Philadelphia_Eagles_logo.svg.png', 
              'Pittsburgh Steelers': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Pittsburgh_Steelers_logo.svg/100px-Pittsburgh_Steelers_logo.svg.png', 
              'Las Vegas Raiders': 'https://upload.wikimedia.org/wikipedia/en/thumb/4/48/Las_Vegas_Raiders_logo.svg/100px-Las_Vegas_Raiders_logo.svg.png', 
              'Los Angeles Rams': 'https://upload.wikimedia.org/wikipedia/en/thumb/8/8a/Los_Angeles_Rams_logo.svg/100px-Los_Angeles_Rams_logo.svg.png',
              'Los Angeles Chargers': 'https://upload.wikimedia.org/wikipedia/en/thumb/7/72/NFL_Chargers_logo.svg/100px-NFL_Chargers_logo.svg.png', 
              'Seattle Seahawks': 'https://upload.wikimedia.org/wikipedia/en/thumb/8/8e/Seattle_Seahawks_logo.svg/100px-Seattle_Seahawks_logo.svg.png', 
              'San Francisco 49ers': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/San_Francisco_49ers_logo.svg/100px-San_Francisco_49ers_logo.svg.png', 
              'Tampa Bay Buccaneers': 'https://upload.wikimedia.org/wikipedia/en/thumb/a/a2/Tampa_Bay_Buccaneers_logo.svg/100px-Tampa_Bay_Buccaneers_logo.svg.png', 
              'Washington Football Team': 'http://www.nflgamedata.com/was.png'}

reverse_logos = dict(zip(logos_dict.values(), logos_dict.keys()))

#boilerplate function to display each linked image in HTML
def path_to_image_html(path):
    return '<img src="'+ path + '" width="35" >'  
    # + " "  + pd.Series(path).map(reverse_logos).to_string().replace("0", "")

#creating new df and mapping links
logo_df = finished_df.copy().reset_index(drop=True)
logo_df['Home_Team'] = logo_df['Home_Team'].map(logos_dict)
logo_df['Away_Team'] = logo_df['Away_Team'].map(logos_dict)

#displays the dataframe as an HTML object
st.markdown(logo_df.to_html(escape=False, formatters=dict(Home_Team=path_to_image_html,  Away_Team=path_to_image_html)), unsafe_allow_html=True)

# pd.set_option('display.max_columns', None)
# print(finished_df)