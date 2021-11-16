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
driver = webdriver.Chrome('chromedriver', options=chrome_options)

#update teams correctly 'rai', 'rav',
teams = ['htx', 'kan', 'nyj', 'buf', 'sea', 'atl', 'phi', 'was', 'cle',  'mia', 'nwe', 'gnb', 'min', 'clt', 'jax', 'chi', 'det', 'car', 'sdg', 'cin', 'crd', 'sfo', 'tam', 'nor', 'dal', 'ram', 'pit', 'nyg', 'oti', 'den', 'rai', 'rav']

no_table = []
url = 'https://www.pro-football-reference.com'
year = 2021

for team in teams:
  driver.get(url + '/teams/' + str(team) + '/2021.htm') 
  sleep(randint(2,10))
  table = pd.read_html(driver.page_source)
  # week = 20
  cols = ['Week', 'Day', 'Date', 'Time', 'BoxS', 'Result', 'OT',	'Rec', 'Home', 'Opp_Name',	'Tm',	'Opp',	'OFF1stD',	'OFFTotYd',	'OFFPassY',	'OFFRushY',	'TOOFF',	'DEF1stD',	'DEFTotYd',	'DEFPassY',	'DEFRushY',	'TODEF',	'OffenseEP',	'DefenseEP',	'Sp_TmsEP']
  dft = table[2]
  dft.columns = dft.columns.droplevel()
  dft.columns = cols
  dft.dropna(subset=['Opp_Name'], inplace=True)
  dft = dft[~dft.Opp_Name.str.contains("Bye", na=False)]
  dft = dft[~dft.Date.str.contains("Playoffs", na=False)]
  dft = dft.drop(['Day', 'Time', 'BoxS', 'OT', 'Rec'], axis=1)
  dft['Result'] = [0 if r=='L' else 1 for r in dft['Result']]
  dft['Home'] = [0 if r=='@' else 1 for r in dft['Home']]
  dft['TOOFF'] = dft['TOOFF'].fillna(0)
  dft['TODEF'] = dft['TODEF'].fillna(0)
  dft['OffenseEP'] = dft['OffenseEP'].fillna(0)
  dft['DefenseEP'] = dft['DefenseEP'].fillna(0)
  dft['Sp_TmsEP'] = dft['Sp_TmsEP'].fillna(0)
  dft['Team'] = str(team)
  dft = dft.set_index('Team')
  dft.reset_index(inplace=True)
  no_table.append(dft)


driver.close()   
df = pd.concat(no_table)

df.to_csv('D:/Documents/ML_DOCS/ML Project Files/2021-NFL-Model/NFL-Score-Prediction/NFL21_update_full.csv')