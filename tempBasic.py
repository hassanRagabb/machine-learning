# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

x = 6+6

print(x)


for x in range(2, 6):
    print(x)

for i in range(10, 1, -2):
    print(i)

print(" friutssssssss")


fruits = ["apple", "banana", "cherry"]
print("fruits[1:4]", fruits[0:2])
for x in fruits:
    print(x)

print(" \n\nnested loop with list")
Adj = ["red", "green", "blue"]
fruits = ["apple", "banana", "cherry"]
for x in Adj:
    for y in fruits:
        print(x, y)


print(" \n\ndraw")
list = [1, 5, 6]
histo = '*'
for x in range(len(list)):
    print(list[x]*histo)


print(" \n\ndictionary")
dict = {"name": "Hassan", "age": 21}
print(dict["name"])


print(" \n\ndictionary")


def hello(fname, lname="ragab"):
    print("hello", " ", fname, " ", lname)


hello("hassan")


print(" \n\n list as parameter")


def funTakeList(fruite):
    for x in fruite:
        print(x)


fruite = ["orange", "banana", "appel"]
funTakeList(fruite)


print(" \n\n classesssssssssssssssssssssssssssssssssssssssssssssssssssssss oop")


class dummyclass:
    x = 20
    y = 10

    def summy(self):
        print(self.x+self.y)

    def div(self):
        print(self.x/self.y)


obj1 = dummyclass()
obj1.summy()
obj1.div()



#oop classes


print(" \n\n classesssssssssssssssssssssssssssssssssssssssssssssssssssssss constructor")
#constructor
class Person:
    def __init__(self, name, age):#constructor
        self.name = name
        self.age = age
        
    
    def myfunc(self):
        print("Hello my name is " + self.name)
    



p1 = Person("hassan", 21)#init will work as constructor
print(p1.name,"\n")
p1.myfunc()




print(" \n\n classesssssssssssssssssssssssssssssssssssssssssssssssssssssss Destructor")
#oop classes
#Destructor object
class child:
  def __init__(self, name, age):#constructor
        self.name = name
        self.age = age
  def __del__(self):
      print(self.__class__.__name__,"destroyed")

  def myfunc(self):
    print("Hello my name is " + self.name)

childd = child("child", 0)
childd.myfunc()

del childd












































































