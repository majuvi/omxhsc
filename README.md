## OMXH Small Cap Dashboard

Download anaconda: https://www.anaconda.com/

Set up Python environment, add it to Jupyter notebook, launch Jupyter and select this environment
```
conda install --name inderes --file requirements.txt
python -m ipykernel install --user --name inderes --display-name "Python (inderes)"
```

Fill your login details in *secrets.json*. You need Inderes premium to download all PDFs:
```
{
    "INDERES_LOGIN_EMAIL" : "",
    "INDERES_LOGIN_PASSWORD" : "",
    "KAUPPALEHTI_LOGIN_EMAIL" : "",
    "KAUPPALEHTI_LOGIN_PASSWORD" : ""
}
```

Download chromedriver: https://googlechromelabs.github.io/chrome-for-testing/


Run script [inderes.ipynb](inderes.ipynb) to fetch recommendations. 
* The script will download all events from https://inderes.fi front page into *frontpage.csv*.
* The script will then download all PDFs from events tagged with "Analyysi" into *reports/*.
* These PDFs are parsed for both recommendation and analyst into *reports_info/analysis_unfixed.csv*.
* The script will probe you to add a few manual fixes to CSV files created in *reports_info/*.
* The final recommendations file is saved as *analysis.csv*.

Run script [kauppalehti.ipynb](kauppalehti.ipynb) to fetch stock prices and dividends. 
* The script will download all historical stock prices and dividends from https://kauppalehti.fi 
* The stock prices and dividends are saved into *dls/* using the stock ticker as the file name.
* Total returns are calculated by reinvesting dividends at the closing price of the ex-dividend date.
* The final historical return file is saved as *prices.csv*.

If new stocks are listed or Inderes starts to follow new companies, they need to be manually added to *key_merge.csv* in order to link the two data sets with inderes-tag and Kauppalehti/Nasdaq ticker. An automatically created file *key_kauppalehti.csv* will link the ticker into the company name.

Run script [recommendations_analysis.ipynb](recommendations_analysis.ipynb) to analyze recommendation performance
* The script will use the *analysis.csv* and *prices.csv* files to print and visualize statistics.
* The script will save files in *assets/* that contain pre-calculated statistics required for the dashboard.

Run the [app.py](app.py) to launch the dashboard and visit http://127.0.0.1:8050/ 
```
python app.py 
```
The app can be easily deployed to Heroku, for example.