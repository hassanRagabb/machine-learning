""" import pandas as pd

df = pd.read_csv('data.csv')

print(df.to_string()) 

"""

    


                    #Pandas Dataframs 
print("\n\nDataframs ");          
import pandas as pd
#dectionary
mydataset = {
  'cars': ["BMW", "Volvo", "Ford"],
  'passings': [3, 7, 2]
}

myvar = pd.DataFrame(mydataset) #dataframe will make the data in table



print(myvar)
print("\n\nDataframs row index");
print(myvar.loc[0])
print("\n\nDataframs list of indexes:");
print(myvar.loc[[0, 1]])




                     #Pandas series  
print("\n\nseries ");
import pandas as pd

a = [1, 7, 2]

myvar = pd.Series(a)

print(myvar)
print(myvar[0])

#Pandas series change index in rows to label 
print("\n\nseries with label");
import pandas as pd

a = [1, 7, 2]

myvar = pd.Series(a, index = ["x", "y", "z"])

print(myvar)
print(myvar["y"])

print("\n\nseries with dictionary");
import pandas as pd

calories = {"day1": 420, "day2": 380, "day3": 390}

myvar = pd.Series(calories)

print(myvar)
import pandas as pd

calories = {"day1": 420, "day2": 380, "day3": 390}

myvar = pd.Series(calories, index = ["day1", "day2"])

print("\n\nseries with part of dictionary");
print(myvar)


print("\n\nReadddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd csv")
print("\n\nRead csv")
import pandas as pd

df = pd.read_csv('data.csv')

print(df.to_string()) 
print("\n\nRead csv only 5 rows first and 5 rows from end")
import pandas as pd

df = pd.read_csv('data.csv')

print(df) 



print("\n\nRead json ")
import pandas as pd

df = pd.read_json('data.json')

print(df.to_string()) 

print("\n\nDectionary dataframe as  json ")

import pandas as pd

data = {
  "Duration":{
    "0":60,
    "1":60,
    "2":60,
    "3":45,
    "4":45,
    "5":60
  },
  "Pulse":{
    "0":110,
    "1":117,
    "2":103,
    "3":109,
    "4":117,
    "5":102
  },
  "Maxpulse":{
    "0":130,
    "1":145,
    "2":135,
    "3":175,
    "4":148,
    "5":127
  },
  "Calories":{
    "0":409,
    "1":479,
    "2":340,
    "3":282,
    "4":406,
    "5":300
  }
}

df = pd.DataFrame(data)

print(df) 

print("\n\n\n\n\n\n\n\n\n\nanalysing data ")

import pandas as pd

df = pd.read_csv('data.csv')
print("\n\nget first 10 row data ")
print(df.head(10))

print("\n\nget end 10 row data ")
print(df.tail(10))


print("\n\ninformation about data.csv ")
print(df.info()) 

            

                             

    




    
    
                                        #pandas dataset cleaning
 

print("\n\n\n\n\n\n\n\n\n\ndataset cleaning ")                                    




#Empty Cells
print("\n\nEmpty Cells this code Return a new Data Frame with no empty cells:")
import pandas as pd

df = pd.read_csv('data.csv')

new_df = df.dropna()#drop row that have  null value  but the original df still have null value if u need original not have null look bottom

print(new_df.to_string())
print(new_df.info()) 
print("\n\ndf original not have null now:")
import pandas as pd

df = pd.read_csv('data.csv')

df.dropna(inplace = True)

print(df.to_string())
print(df.info()) 














print("\n\nReplace all cols NULl empty cells values with the number 130 :")

import pandas as pd

df = pd.read_csv('data.csv')

df.fillna(130, inplace = True)
print(df.to_string())
print(df.info()) 






print("\n\n\n\nReplace specific  column that have NULl empty cells values with the number 130 :")
import pandas as pd

df = pd.read_csv('data.csv')

df["Calories"].fillna(130, inplace = True)
print(df.to_string())





print("\n\nCalculate the MEAN(avg), and replace any empty values with it: median() sort them and take the middel element  mode()[0] the more repeated will take it:")
import pandas as pd

df = pd.read_csv('data.csv')

x = df["Calories"].mean()

df["Calories"].fillna(x, inplace = True)
print(df.to_string())






print("\n\nReplace date  column that have wrng  format like 20220927  will become '2022/09/27'  but null value will change to NAT SO DROP null data:")
import pandas as pd

#df = pd.read_csv('data.csv')

# catch colmn that have att Daate using-->    df['Date'] = pd.to_datetime(df['Date'])

#print(df.to_string())

print("\n\nDrop row that have null date ")
#df.dropna(subset=['Date'], inplace = True)


 
print("\n\nwrong values all rows have value may be 50 55 60 53 like that but one row have value big like 555 ")
df.loc[7, 'Duration'] = 45 #go to row 7 in column Duration that have big value and chnage this value to 45 u can make a loop for this column



print("\n\n\n\n\n\nwrong values Loop through all values in the Duration column.If the value is higher than 120, set it to 120: ")
for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.loc[x, "Duration"] = 120


print("\n\n\n\n\n\nremoving duplicate for solve ovverfiting values(duplicate value not good in machine learning )")
print(df.duplicated())
#or
df.drop_duplicates(inplace = True) # drop in all cols









print("\n\n\n\n\n\n correlation is the relationship between features in dataset: ")
print("\n\n\n\n\n\nShow the relationship between the columns: ")
df.corr()
#1.000 is good corr
#big than(0.6 or( -0.6 like -0.9) is good also
#bad corre is less than 0.6 like 0.00034

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

df.plot()

plt.show()

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

df.plot(kind = 'scatter', x = 'Duration', y = 'Calories')

plt.show()



import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

df.plot(kind = 'scatter', x = 'Duration', y = 'Maxpulse')

plt.show()






df["Duration"].plot(kind = 'hist')




