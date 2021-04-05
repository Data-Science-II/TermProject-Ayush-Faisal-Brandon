import requests
import zipfile
import os 

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
    
if __name__ == '__main__':
    for year in [2014, 2015, 2016, 2017]: 
        download_and_unzip(year)


