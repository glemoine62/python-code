import requests
import numpy as np
from datetime import datetime
from pandas import DataFrame


def getKNMISeries(bdate, edate, stn, params):

    b = bdate.split('-')
    e = edate.split('-')
    payload = {'lang': 'nl', 'byear': b[0], 'bmonth': b[1], 'bday': b[2], 
           'eyear': e[0], 'emonth': e[1], 'eday': e[2], 
           'stations': stn, 'variabele': params}

    r = requests.get('http://projects.knmi.nl/klimatologie/daggegevens/getdata_dag.cgi', params=payload)

    lines = r.content.split('\r\n')

    rows = []
    for l in lines:
        if len(l) > 0 and not l.startswith('#'):
            cols = l.split(',')[1:len(params)+2]
        
            cols[0] = datetime.strptime(cols[0], '%Y%m%d')  # date field
            for i in range(0,len(params)):
                if len(cols[i+1].strip()) != 0:
                    cols[i+1] = int(cols[i+1])
                else:
                    cols[i+1] = np.nan

            rows.append(cols)

    df = DataFrame(rows, columns = ['date'] + params)

    df.set_index(['date'], inplace=True)
    return df
