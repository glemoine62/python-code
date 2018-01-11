import sys
import pandas as pd
import numpy as np

df = pd.read_csv(sys.argv[1])

threshold = 0.50

dim = len(df.columns)-2
r_index = df.columns[2:]

cm = np.zeros((dim, dim))

for i in df.index:
	row = df.loc[i]
	maxprob = np.argmax(row[r_index])
	n = r_index.tolist().index(maxprob)
	cm[row.klass][n] += 1 

np.set_printoptions(suppress=True, precision=0)

print cm.astype(int)