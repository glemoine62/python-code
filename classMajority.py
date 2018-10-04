import pandas as pd
import numpy as np

df = pd.read_csv('DK2017_cropselect_classes.csv', index_col = 0)
#df = pd.read_csv('test.csv', index_col = 0)

r_index = df.columns[1:]

def majority(s):
    v, c = np.unique(s[r_index], return_counts=True)   
    return pd.Series({'majarg': v[np.argmax(c)], 'majcount': np.max(c)})

df = df.merge(df.apply(majority, axis=1), left_index=True, right_index=True)

print "Error rate ", 100.0*len(df[df['klass']!=df['majarg']])/float(len(df))

df.to_csv('DKtest.csv')