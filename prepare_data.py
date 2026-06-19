import pandas as pd
print('Loading full dataset...')
df = pd.read_csv('data/raw/criteo_uplift.csv')

print('Creating balanced 500k sample...')
treated = df[df['treatment']==1].sample(n=250000, random_state=42)
control = df[df['treatment']==0].sample(n=250000, random_state=42)
sample = pd.concat([treated, control]).sample(frac=1, random_state=42).reset_index(drop=True)

sample.to_csv('data/processed/criteo_sample.csv', index=False)
print('Saved! Shape:', sample.shape)
print('Treatment ratio:', sample['treatment'].mean())
print('Conversion rate:', sample['conversion'].mean())
print('Columns:', sample.columns.tolist())
