#!/usr/bin/env python
# coding: utf-8

# CSCI 4360 - Data Science II Term Project
# Author: Ayush Kumar, Brandon Amirouche, Faisal Hossain

# Import Libraries
import os 
import quandl
import zipfile
import sqlite3
import requests
import pandas as pd

def download_and_unzip(year): 
    """
    Downloads data from the HMDA website for a given year.
    Keyword arguments: 

    year - the year of data to be downloaded
    """

    url = f"https://files.consumerfinance.gov/hmda-historic-loan-data/hmda_{year}_nationwide_all-records_labels.zip"
    datafile = requests.get(url, allow_redirects = True)
    
    filename = f"{year}.zip"
    print(filename)
    
    open(filename, 'wb').write(datafile.content)
    
    with zipfile.ZipFile(filename, 'r') as zip_ref: 
        zip_ref.extractall('.')
        
    os.remove(filename)
    
    # Creating SQLDatabase 
    f = open("HMDA.db", "w+")
    f.close()

    db = sqlite3.connect("HMDA.db")
    csv = f"hmda_{year}_nationwide_all-records_labels.csv"
    for chunk in pd.read_csv(csv, chunksize=100000, low_memory=False): 
        chunk.to_sql(f"hmda_{year}", db, if_exists = "append")
    
    db.close()

def zillow_download(years):
    """
    Downloads Zillow Real Estate data from the quandl website for a given year.

    Keyword arguments: 
    years - list of years of data to be downloaded
    """

    # Authentication
    quandl.ApiConfig.api_key = "koRtyVvbCo8Z-rArzJny"

    # API request to get data
    zillow_data = quandl.get_table('ZILLOW/DATA')
    zillow_indicators = quandl.get_table('ZILLOW/INDICATORS')
    zillow_regions = quandl.get_table('ZILLOW/REGIONS')

    # Merge Data Frames by 'indicator_id'
    zillow_data_indicators = pd.merge(zillow_data, zillow_indicators, on='indicator_id')
    # Merge Data Frames by 'region_id'
    zillow3 = pd.merge(zillow_data_indicators, zillow_regions, on='region_id')

    # Download Merged Dataset as CSV
    cwd = os.getcwd() + "/zillow_merged.csv"
    zillow3.to_csv (cwd, index = False, header=True)

    # Filter by year and Dowload as CSV 
    for year in years:
        df = zillow3[(zillow3['date'] > f"{year}-01-01") & (zillow3['date'] < f"{year}-12-31")]
        cwd = os.getcwd()
        cwd += f"/zillow_{year}.csv"
        df.to_csv (cwd, index = False, header=True)


if __name__ == '__main__':
    for year in [2014, 2015, 2016, 2017]: 
        download_and_unzip(year)

    # Dowload Zillow data from 2014-2019 in CSV format
    zillow_download(list(range(2014, 2019)))
