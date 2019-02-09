import pandas as pd
import os
import numpy as np

def create_csv(current_month_days):
    PATH_ATT = 'Dataset'
    att = pd.DataFrame(None, columns=['Date'])

    for f in os.listdir(PATH_ATT):

        # print(f)
        att[f] = ""
        att= pd.DataFrame(data=0*np.ones((current_month_days,len(att.columns))), columns=list(att.columns))
        # df.to_csv(current_month + '.csv')
    return att