# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 14:13:53 2022

@author: h
"""

                                         

        
                                    #matplotlib tutorial 
#https://www.youtube.com/watch?v=tTvemyJlSJI&list=PLPBnj6azlABak3muRtjhHavcO62p-bFhU&index=11

import matplotlib.pyplot as plt
import numpy as np
# x have 2 value
xpoints = np.array([0, 6])
ypoints = np.array([0, 250])

plt.plot(xpoints, ypoints)
plt.show()



# x have 3 value
import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([0, 6,10])
ypoints = np.array([0, 250,350])

plt.plot(xpoints, ypoints)
plt.show()



# not draw line only circle
import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([1, 8])
ypoints = np.array([3, 10])

plt.plot(xpoints, ypoints, 'o') #look at line 59
plt.show()






#only one array will assume as default  ( xAxiss is [0,1,2,3,4,5....] and the input are y)
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10, 5, 7])

plt.plot(ypoints)
plt.show()




                 #marker
 #dot line with circle
import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, 'o:r',ms=20,mec='b')#'o:r' o means circle and r for draw line #ms is the size #mec is round line above the circle here color azrq in cirlce 
plt.show()






# x axis name and y axis name 
#Matplotlib Labels and Title
import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.plot(x, y)

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.show()







##########
import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}

plt.title("Sports Watch Data", fontdict = font1)
plt.xlabel("Average Pulse", fontdict = font2)
plt.ylabel("Calorie Burnage", fontdict = font2)

plt.plot(x, y)
plt.show()



                        # grid only background lines note math 
                        
import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)

plt.grid(axis = 'x')

plt.show()                        
                    
                    
                    
                    
                    
                    # subplots need to draw many shapes besides thogether till 4 or many in 4 u have picture 2 rows and 2 cols 
#plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(1, 2, 1)# (row,cols,first)
plt.plot(x,y)

#plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(1, 2, 2)
plt.plot(x,y)

plt.show()





                #scater
import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

plt.scatter(x, y)# many points 
plt.show() 

                #bar
x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.bar(x,y)
plt.show()




                 #histogram

   # get 250 value in range (170) using standared deviation 10 
x = np.random.normal(170, 10, 250)
print(x)
    #Draw Histogram
x = np.random.normal(170, 10, 250) # random form zero to 169 using standerddeviation 10
plt.hist(x)
plt.show() 
"""
            READ hIstogram shape
2 people from 140 to 145cm
5 people from 145 to 150cm
15 people from 151 to 156cm
31 people from 157 to 162cm
46 people from 163 to 168cm
53 people from 168 to 173cm
45 people from 173 to 178cm
28 people from 179 to 184cm
21 people from 185 to 190cm
4 people from 190 to 195cm

"""
188


              
#  Pie Charts    for stat in sales something 

y = np.array([35, 25, 25, 15])

plt.pie(y)
plt.show() 


















