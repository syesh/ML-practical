import numpy as np
import pandas as pd
data = pd.read_csv('/content/enjoysport.csv - Sheet1.csv')
attribute = np.array(data)[:,:-1]
target = np.array(data)[:,-1]
print("Training Data: \n")
print(data)
def train(att,tar):
  for i,val in enumerate(tar):
    if val == 'yes':
      specific_h = att[i].copy()
      break
  for i,val in enumerate(att):
    if tar[i] == 'yes':
      for x in range(len(specific_h)):
        if val[x]!=specific_h[x]:
          specific_h[x]='?'
        else:
          pass
  return specific_h
print("\n")
print(train(attribute,target))
