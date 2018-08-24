import pandas as pd
import numpy as np
import os

relativePath = os.getcwd()
path = relativePath + "/Results/08082018-21/"

df = pd.read_csv(path+"predicted_result.csv", index_col="Sentence#")
jump = int((len(df.columns)+1)/2)
print(jump)
print(df.columns[jump-1])
#print(df.columns[5], "   ", df.columns[5+jump])

values=[]

for j in range(jump-1):
    yesyes=0
    yesno=0
    noyes=0
    nono=0
    a=df[df.columns[j]].values
    b=df[df.columns[j+jump]].values
    for i in range(len(a)):
        if a[i]==1:
            if b[i]==1:
                yesyes+=1
            else:
                yesno+=1
        else:
            if b[i]==1:
                noyes+=1
            else:
                nono+=1
    total = yesyes+yesno+noyes+nono
    values.append([df.columns[j],total,yesyes,yesno,noyes,nono, yesyes/total, yesno/total, noyes/total, nono/total])

result = pd.DataFrame(values, columns=["subtopic", "Total","YesYes", "YesNo", "NoYes", "NoNo", "YesYes%", "YesNo%", "NoYes%", "NoNo%"])
result.to_csv(path + 'predicted_result_summary.csv')