# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:39:50 2022

@author: h
"""

import numpy as np
#from numpu import *


print("1d array")
arr = np.array([1, 2, 3, 4], float)
print(arr)
print(type(arr))
print(arr[:2])  # index 2 not join


print("2d array")
twoDArray = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], float)
print(twoDArray)
print(type(twoDArray))
print("#get all rows row 2 not join start from num 0")
print(twoDArray[:2])  # get all rows row 2 not join
print("row number 2 with all cols in it")
print(twoDArray[1, :])  # get row number 2

print("get cell 3 from all rows")
# get the index 2 start from 0 1 [2] this index from  all rows
print(twoDArray[:, 2])
print("get from the last row from cell 2 mn end to the end")
print(twoDArray[-1:, -2:])  # print(twoDArray[-1,-2:])
print("get from the before last row from cell 2 as start to end ")
print(twoDArray[-2:, -2:])


print("shape proberty for array return tuple with size of each array dimention")
print(twoDArray.shape)  # [2,4]
print(len(twoDArray))  # 2

print("dtype proberty tell u the type of values are stored in it")
print(twoDArray.dtype)  # float64


print("check for value inside array")
print(2 in twoDArray)  # true
print(0 in twoDArray)  # false


print("make array froom index 0 to 9 9 will join ")
a = np.array(range(10), float)
print(a)
b = a  # if u change in a auto this change will affect b

c = a.copy()  # if u change in a  this change will not affect c

print("reshape array  to td array as 5 rows and 2 cols ")
a = a.reshape((5, 2))
print(a)
print(a.shape)#(5, 2)


# Lists can also be created from arrays:
a = np.array([1, 2, 3], float)
a.tolist()  # [1.0, 2.0, 3.0]
print(list(a))  # [1.0, 2.0, 3.0]



            #change to string noone understand it binary and 
a = np.array([1, 2, 3], float)
s = a.tostring()
print(s)#'\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@'
print(np.fromstring(s))#array([ 1., 2., 3.])






print("\n\nOne can fill an array with a single value:")
a =np.array([1, 2, 3], float)
print(a)#array([ 1., 2., 3.])
a.fill(0)
print(a)#array([ 0., 0., 0.])





print("\n\n\ntranspose row to become column")
a = np.array(range(6), float).reshape((2, 3))
print(a)           #array([[ 0., 1., 2.],
                          #[ 3., 4., 5.]])
                          
print(a.transpose())#array([[ 0., 3.],
                          # [ 1., 4.],
                          #[ 2., 5.]])


print("\n\n\nchange 2d to 1d")
a = np.array([[1, 2, 3], [4, 5, 6]], float)
print(a)#array([[ 1., 2., 3.],
                #[ 4., 5., 6.]])
print(a.flatten())#array([ 1., 2., 3., 4., 5., 6.])



print("\n\n\nconcatenate more 1d arrays to become ONE ID array AT ENd")
a = np.array([1,2], float)
b = np.array([3,4,5,6], float)
c = np.array([7,8,9], float)
print(np.concatenate((a, b, c)))#array([1., 2., 3., 4., 5., 6., 7., 8., 9.])






print("\n\n\nconcatenate more 2d arrays to become one 2d array ")
a = np.array([[1, 2], [3, 4]], float)
b = np.array([[5, 6], [7,8]], float)
print(np.concatenate((a,b)))#array([[ 1., 2.],
                                   #[ 3., 4.],
                                   #[ 5., 6.],
                                   #[ 7., 8.]])
print("\n\n finish a then add b when u finish a ")                              
print(np.concatenate((a,b), axis=0))#array([[ 1., 2.],
                                            #[ 3., 4.],
                                            #[ 5., 6.],
                                            #[ 7., 8.]])
                                            
                                         
print("\n\n1 finish a then add b as rows in a")                          
print(np.concatenate((a,b), axis=1))#array([[ 1., 2., 5., 6.],
                                            #[ 3., 4., 7., 8.]])




 #explain  axis=0 in only one array  line 300







print("solve shape when give u onlu the cols not rows ")
a = np.array([1, 2, 3], float)
print(a.shape)#give u only cols not rows
print("\n\nonly one column with many rows")
print(a[:,np.newaxis])#array([[ 1.],
                            # [ 2.],
                            #[ 3.]])
print("\n\nshape now")   
print(a[:,np.newaxis].shape)#(3,1)
print("\n\nERROR only one row with many columns")
print(b[np.newaxis,:])#array([[ 1., 2., 3.]])
print("\n\n")
print(b,"b shape is", b[np.newaxis,:].shape)#(1, 2, 2)



c=a[np.newaxis,:]
print("\n\n",c,"c shape is:",c.shape)

print("\n\nOther ways to create arrays")
print("\n\n1- function arange similar to the range function but returns an array:")
a=np.arange(5, dtype=float)#array([ 0., 1., 2., 3., 4.])
print("a is",a)
b=np.arange(1, 6, 2, dtype=int)#array([1, 3, 5]) here  start =1 , end are 6 ,and ++2 
print("b is",b)

print("\n\n2- function zero and one take only dimention and all values are ones or zeros")
print(np.ones((2,3), dtype=float))#array([[ 1., 1., 1.],
                                    # [ 1., 1., 1.]])
print(np.zeros(7, dtype=int))#array([0, 0, 0, 0, 0, 0, 0])

print("\n\t- take dimention from anoter array")
a = np.array([[1, 2, 3], [4, 5, 6]], float)
print(np.zeros_like(a),"\n")#array([[ 0., 0., 0.],
                             #[ 0., 0., 0.]])
print(np.ones_like(a))#array([[ 1., 1., 1.],
                             #[ 1., 1., 1.]])


print("\n\n diagonal =1 and other are zeros")
z=np.identity(4, dtype=float)
print(z)
d=np.eye(4, k=1, dtype=float)
print("\n",d)











                                                             #Array mathematics
        
a = np.array([1,2,3], float)
b = np.array([5,2,6], float)
print("\n\noperation in arrays",)
print("+",a + b)#array([6., 4., 9.])
print("-",a - b)#array([-4., 0., -3.])
print("*",a * b)#array([5., 4., 18.])
print("/",b / a)#array([5., 1., 2.])
print("%",a % b)#array([1., 0., 3.])
print("base ** power",b**a)#rray([5., 4., 216.])


print("\n\nmulti 2 arrays 2d")
a = np.array([[1,2], [3,4]], float)
b = np.array([[2,0], [1,3]], float)
print( a * b)#array([[2., 0.], [3., 12.]])


print("\n\n ERROR sum 2 arrays 1d check comments 4 lines ")
a = np.array([1,2,3], float)
b = np.array([4,5], float)
#print(a + b)#shape mismatch
#Traceback (most recent call last):
 #File "<stdin>", line 1, in <module>
#ValueError: shape mismatch: objects cannot be broadcast to a single shape

print("\n\n DONE sum 2 arrays 2d using broadcating method")
a = np.array([[1, 2], [3, 4], [5, 6]], float)
b = np.array([-1, 3], float)
print("\na",a)

print("\nb",b)
print(" \na + b", a + b)


a = np.zeros((2,2), float)
b = np.array([-1., 3.], float)
print("\n\n",a)

print( b)

print("\n a + b",a + b)

print(a + b[np.newaxis,:])
print( a + b[:,np.newaxis])



a = np.array([1, 4, 9], float)
print("\n\n",np.sqrt(a))


a = np.array([1.1, 1.5, 1.9], float)
print("\n\n",np.floor(a))
print("\n\n",np.ceil(a))
print("\n\n",np.rint(a))


print("\n pi=",np.pi)

print("\n e=",np.e)



        #array iteration

a = np.array([1, 4, 5], int)
print("\n\narray values")   
for x in a:
    print(x)
print("rows using  row by row")   
a = np.array([[1, 2], [3, 4], [5, 6]], float)
for x in a:
    print(x)
print("multiply rows using  product row by row")   
a = np.array([[1, 2], [3, 4], [5, 6]], float)
for (x, y) in a:
  print( x * y)


print("sum array and product ")
a = np.array([2, 4, 3], float)
print(a.sum())#9.0
print(a.prod())#24.0

print(np.sum(a))#9.0
print(np.prod(a))#24.0


print("\nmimimum",a.min())

print("max",a.max())

a = np.array([2, 1, 9], float)
print("mean middle",a.mean())
print("variance", a.var())
print("std",a.std())



a = np.array([2, 1, 9], float)
print("\n\nindex for small value",a.argmin())
print("index max", a.argmax())


a = np.array([[0, 2], [3, -1], [3, 5]], float)
print("\n\nwork with first element in rows together and second element in rows also together:",a.mean(axis=0))
#array([ 2., 2.])
print("average for each row togther" ,a.mean(axis=1))
#array([ 1., 1., 4.])
print("min in each row",a.min(axis=1))
#array([ 0., -1., 3.])
print(a.max(axis=0))
#array([ 3., 5.])
 


print("\n\nLike lists, arrays can be sorted:")
a = np.array([6, 2, 5, -1, 0], float)
print( "\nsorted(a)",sorted(a))
#[-1.0, 0.0, 2.0, 5.0, 6.0]
print("a.sort",a.sort())
print("a", a)
#array([-1., 0., 2., 5., 6.])


print("\n\nrearrange values in arrays by range ")
a = np.array([6, 2, 5, -1, 0], float)
print(a.clip(0, 5))#array([ 5., 2., 5., 0., 0.]) using this method  min(max(x, minval), maxval) x is a and minval is 0 and maxval is 5 take them as parameter


a = np.array([1, 1, 4, 5, 5, 5, 7], float)
print("\nonly one sample machine learning",np.unique(a))#array([ 1., 4., 5., 7.])



a = np.array([[1, 2], [3, 4]], float)
print("\ndiagonal",a.diagonal())#array([ 1., 4.])



                                    #Comparison operators and value testing
print( "\n\n\n\nComparison operators and value testing")

a = np.array([1, 3, 0], float)
b = np.array([0, 3, 2], float)
print( "a>b",a > b)#array([ True, False, False], dtype=bool)

print(  "a==b",a == b)
print(  "a <= b", a <= b)
print(  "a > 2",  a > 2)


c = np.array([ True, False, False], bool)
print("any one be true",any(c))
print("all must be true else false", all(c))



a = np.array([1, 3, 0], float)
print("\n\nlogical and", np.logical_and(a > 0, a < 3))#array([ True, False, False], dtype=bool)
b = np.array([True, False, True], bool)
print(np.logical_not(b))#array([False, True, False], dtype=bool)
c = np.array([False, True, False], bool)
print( np.logical_or(b, c))#array([ True, True, True], dtype=bool)



a = np.array([1, 3, 0], float)
print("\n\ndivide each element in array using (1\element inside array)",np.where(a != 0, 1 / a, a))
#array([ 1. , 0.33333333, 0. ])

print("\nif true put 3 else false put 2" ,np.where(a > 0, 3, 2))


a = np.array([[0, 1], [3, 0]], float)
print("\n\nindex if nonzero print 1 if zero print zero", a.nonzero())
#(array([0, 1]), array([1, 0]))


a = np.array([1, np.NaN, np.Inf], float)
print("\n\na",a)#array([ 1., NaN, Inf])
print("is nan",np.isnan(a))#array([False, True, False], dtype=bool)
print("is finite like numbers",np.isfinite(a))#array([ True, False, False], dtype=bool)



a = np.array([[6, 4], [5, 9]], float)
print("\n\nonly true false",a >= 6)#array([[ True, False],
                                         #[False, True]], dtype=bool)
print("\n\nvalues that satisfy this condition ",a[a >= 6])#array([ 6., 9.])

print("\n\nvalues >5 and less than 9 are:",a[np.logical_and(a > 5, a < 9)])



a = np.array([2, 4, 6, 8], float)
b = np.array([0, 0, 1, 3, 2, 1], int)
print("\n\nvalue of a using b as index",a[b])#array([ 2., 2., 4., 8., 6., 4.])
print("same",a[[0, 0, 1, 3, 2, 1]])


a = np.array([[1, 4], [9, 16]], float)
b = np.array([0, 0, 1, 1, 0], int)
c = np.array([0, 1, 1, 1, 1], int)
print("\n\nvalue of a using b and xas indexes",a[b,c])#array([ 1., 4., 16., 16., 4.])


a = np.array([2, 4, 6, 8], float)
b = np.array([0, 0, 1, 3, 2, 1], int)
print("a[b]",a.take(b))#array([ 2., 2., 4., 8., 6., 4.])


a = np.array([[0, 1], [2, 3]], float)
b = np.array([0, 0, 1], int)
print("\nrow in a depend on index of b",a.take(b, axis=0))#array([[ 0., 1.],
                                                                #[ 0., 1.],
                                                                # [ 2., 3.]])
print("\nprint a using b as cols:",a.take(b, axis=1))#array([[ 0., 0., 1.],
                                                        # [ 2., 2., 3.]])


a = np.array([0, 1, 2, 3, 4, 5], float)
b = np.array([9, 8, 7], float)
a.put([0, 3], b)
print("\n\n\na",a)#array([ 9., 1., 2., 8., 4., 5.])


a = np.array([0, 1, 2, 3, 4, 5], float)
a.put([0, 3], 5)
print("\n\n\na",a)#array([ 5., 1., 2., 5., 4., 5.])


print("\n\n\t\t\t\t\t\t\t mutliply row in cols in matrix")
a = np.array([1, 2, 3], float)
b = np.array([0, 1, 1], float)
print("row *col=",np.dot(a, b))#5.0

a = np.array([[0, 1], [2, 3]], float)
b = np.array([2, 3], float)
c = np.array([[1, 1], [4, 0]], float)
print( "\n\na",a)
#array([[ 0., 1.],
 #[ 2., 3.]])
print("\n\nnp.dot(b, a)",np.dot(b, a))#array([ 6., 11.])
print("\n\nnp.dot(a, b)",np.dot(a, b))#array([ 3., 13.])
print("\n\nnp.dot(a, c)",np.dot(a, c))
#array([[ 4., 0.],
 #[ 14., 2.]])
print("\n\nnp.dot( c,a)",np.dot(c, a))
#array([[ 2., 4.],
 #[ 0., 4.]])




a = np.array([1, 4, 0], float)
b = np.array([2, 2, 1], float)
print("\n\nnp.outer:",np.outer(a, b))
#array([[ 2., 2., 1.],
 #[ 8., 8., 4.],
 #[ 0., 0., 0.]])
print("\nnp.inner like method dot:",np.inner(a, b))#10.0
print("\nnp.cross:",np.cross(a, b))#array([ 4., -1., -6.])


a = np.array([[4, 2, 0], [9, 3, 7], [1, 2, 1]], float)
print("\n\n\na:",a)
print("\nvalue of mo7ded value of matrix :",np.linalg.det(a))#-53.999999999999993