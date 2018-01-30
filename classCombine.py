import pandas as pd

# The join of the first and second run will include ALL parcels,
# because the outer join of df0 and df1 introduce the 20% subset of
# training parcels that were used in run 0 and not in run 1
df0 = pd.read_csv('DK2017_cropselect_0_class.csv', index_col = 0)
df1 = pd.read_csv('DK2017_cropselect_1_class.csv', index_col = 0)

# Create the join and retain 'klass' label as 'klass_1'
df = df0.join(df1, how="outer", rsuffix= '_1')
# Records that were not yet in df0 have 'klass' label missing (NA)
# so, overwrite with those of 'klass_1'
df['klass'].loc[df['klass'].isnull()] = df['klass_1'].loc[df['klass'].isnull()]
# and drop the now redundant 'klass_1' label
df.drop('klass_1', axis=1, inplace=True)

# Load the other runs
df2 = pd.read_csv('DK2017_cropselect_2_class.csv', index_col = 0)
df2.drop('klass', axis=1, inplace=True)
df3 = pd.read_csv('DK2017_cropselect_3_class.csv', index_col = 0)
df3.drop('klass', axis=1, inplace=True)
df4 = pd.read_csv('DK2017_cropselect_4_class.csv', index_col = 0)
df4.drop('klass', axis=1, inplace=True)

# Join the latter, overwrite NA values and save as int
df.join(df2, how="outer").join(df3, how="outer").join(df4, how="outer").fillna(-1).astype(int).to_csv('DK2017_cropselect_classes.csv')