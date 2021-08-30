from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
from random import randint
from selenium import webdriver
from time import sleep

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome('chromedriver', chrome_options=chrome_options)

teams = ['htx', 'kan', 'nyj', 'buf', 'sea', 'atl', 'phi', 'was', 'cle', 'rav', 'mia', 'nwe', 'gnb', 'min', 'clt', 'jax', 'chi', 'det', 'rai', 'car', 'sdg', 'cin', 'crd', 'sfo', 'tam', 'nor', 'dal', 'ram', 'pit', 'nyg', 'oti', 'den']

no_table = []
url = 'https://www.pro-football-reference.com'
year = 2021

for team in teams:
  week = 101
  driver.get(url + '/teams/' + str(team) + '/2021.htm') 
  sleep(randint(2,10))
  table = pd.read_html(driver.page_source)
  cols = ['Week', 'Day', 'Date', 'Time', 'Home',	'Opp_Name', 'Final']
  dft = table[0]
  dft.columns = cols
  dft.dropna(subset=['Opp_Name'], inplace=True)
  dft['Week'] = dft.Week.str.replace("Pre ", "10", regex=True)
  dft = dft[~dft.Opp_Name.str.contains("Preseason", na=False)]
  dft = dft[~dft.Opp_Name.str.contains("Regular Season", na=False)]
  # dft = dft[dft.Week == week]
  dft[['Result','Score']] = dft.Final.str.split(",",expand=True)
  dft[['Tm','Opp']] = dft.Score.str.split("-",expand=True)
  dft['Result'] = [0 if r=='L' else 1 for r in dft['Result']]
  dft['Home'] = [0 if r=='@' else 1 for r in dft['Home']]
  dft = dft.drop(['Day', 'Time', 'Final', 'Score'], axis=1)
  dft['Result'] = dft['Result'].fillna(0)
  dft['Tm'] = dft['Tm'].fillna(0)
  dft['Opp'] = dft['Opp'].fillna(0)
  dft['Team'] = str(team)
  dft = dft.set_index('Team')
  dft.reset_index(inplace=True)
  no_table.append(dft)


driver.close()   
df = pd.concat(no_table)

df['Opp_Name'] = df['Opp_Name'].astype('category')
df['Team'] = df['Team'].astype('category')
df['Week'] = df['Week'].astype('int')
df['Home'] = df['Home'].astype('int')
df['Result'] = df['Result'].astype('int')
df['Tm'] = df['Tm'].astype('int')
df['Opp'] = df['Opp'].astype('int')
df.to_csv('2021df_pretest.csv')