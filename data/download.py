import requests
import zipfile
import os 
import sqlite3
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


if __name__ == '__main__':
    for year in [2014, 2015, 2016, 2017]: 
        download_and_unzip(year)


