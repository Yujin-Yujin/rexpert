import pandas as pd

file_path = "/home/yujin/rexpert/output/analysis/atomic,cwwv/adapter_selection_csqa_12.csv"

df = pd.read_csv(file_path, names=["1","2","3","label"])

print("1option :",df['1'].sum())
print("2option :",df['2'].sum())
print("3option :",df['3'].sum())


print(df)