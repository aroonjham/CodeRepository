##### PYTHON CODE TIPS AND TRICKS

# Comment single line of code using "#"

''' 
to comment multiple lines use :) 
pretty obvious
'''

###########################

##### DATE AND TIME

from datetime import datetime

now = datetime.now()
current_year = str(now.year)
current_month = str(now.month)
current_day = str(now.day)
t_date = current_month+"/"+current_day+"/"+current_year

c_hour = str(now.hour)
c_minute = str(now.minute)
c_second = str(now.second)
t_time = c_hour+":"+c_minute+":"+c_second

print t_date + " " +t_time

############################

##### ASKING FOR USER INPUT + CONVERT USER INPUT INTO LOWERCASE + CHECK IF USER INPUT IS ALL ALPHA

print "Welcome to the English to Pig Latin translator!"
original = raw_input("enter the word you want translated").lower()
if original == "" or original.isalpha() == False:
    print "error"
else:
    print original

############################

##### MORE COMPLEX IF/ELSE AND USE OF LEN FUNCTION

pyg = 'ay'

original = raw_input('Enter a word:')

if len(original) > 0 and original.isalpha():
    word = original.lower()
    first = word[0]
    if first == "a" or first == "e" or first == "o" or first == "i" or first == "u":
        new_word = word+pyg
        print new_word
    else:
        length = len(word)
        new_word = word[1:length]+word[0]+pyg
        print new_word
else:
    print 'empty'
	
############################

##### A SIMPLE FUNCTION

import math
def area_of_circle(radius): #Don't forget to include a : after your if or def statements!
    return math.pi * (radius ** 2)
	
############################

##### SOME MORE FUNCTIONS USING IF/ELSE/ELIF

def hotel_cost(nights):
    return nights * 140

def plane_ride_cost(city): #Don't forget to include a : after your if or def statements!
    if city == "Charlotte":
        return 183
    elif city == "Tampa":
        return 220
    elif city == "Pittsburgh":
        return 222
    elif city == "Los Angeles":
        return 475
    else:
        return ""
        
def rental_car_cost(days):
        if days >= 7:
            return (days*40) - 50
        elif days >= 3:
            return (days*40) - 20
        else:
            return days*40
            
def trip_cost(city, days, spending_money):
    return hotel_cost(days) + plane_ride_cost(city) + rental_car_cost(days) + spending_money
    
print trip_cost("Los Angeles",5,600)

############################

##### INTRODUCTION TO STRINGS AND STRING OPERATORS

myName = "Aroon&Karish&Nihu_R_A_Fam1ly" #example of string"
print myName.center(50)
print myName.count("&")
print myName.split("&")

'''
center	astring.center(w)		Returns a string centered in a field of size w
count	astring.count(item)		Returns the number of occurrences of item in the string
ljust	astring.ljust(w)		Returns a string left-justified in a field of size w
lower	astring.lower()			Returns a string in all lowercase
rjust	astring.rjust(w)		Returns a string right-justified in a field of size w
find	astring.find(item)		Returns the index of the first occurrence of item
split	astring.split(schar)	Splits a string into substrings at schar and returns a LIST

'''

############################

##### INTRODUCTION TO SETS

'''
The common approach to get a unique collection of items is to use a set. Sets are unordered collections of distinct objects. key word is "distinct"
To create a set from any iterable, you can simply pass it to the built-in set() function. 
If you later need a real list again, you can similarly pass the set to the list() function
Elements within a set cannot be modified, but elements can be added to or removed from a set
'''

mystring = 'Aroon Basant Jham'
myset = list(set(mystring))
print myset

'''
methods provided by Sets in Python
'''

aset.union(otherset)	#Returns a new set with all elements from both sets
aset.intersection(otherset)	# Returns a new set with only those elements common to both sets
aset.difference(otherset)	# Returns a new set with all items from first set not in second
aset.issubset(otherset)	# Asks whether all elements of one set are in the other
aset.add(item)	# Adds item to the set
aset.remove(item)	#Removes item from the set
aset.pop()	#Removes an arbitrary element from the set
aset.clear()	#Removes all elements from the set



############################

##### GENTLE INTRODUCTION TO LIST ; REPLACING ELEMENTS IN A LIST ; POINTING TO ELEMENTS IN A LIST ; OTHER COOL THINGS USING LISTS

zoo_animals = ["pangolin", "cassowary", "sloth", "tiger"] ### orinal list

zoo_animals[2] = "hyena" ### replacing the sloth with a hyena

zoo_animals[3] = "lion" ### replacing the tiger in the list with a hyena

suitcase = ["sunglasses", "hat", "passport", "laptop", "suit", "shoes"]

first =    suitcase[0:2]
middle =   suitcase[2:4]
last =     suitcase[4:6]

my_list[:2] # Grabs the first two items
my_list[3:] # Grabs the fourth through last items
my_list[-3:] # Grabs last 3 items in the list

myindex = my_list.index("dog") # .index() will return the first index that contains the string "dog"
my_list.insert(myindex,"cat") # .insert() adds the item "cat" at index "myindex" of my_list, and moves the item previously at index "myindex" and all items following it to the next index 

l = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
print l[2:9:2] # notice the format list[start:end:stride] ... in the example the print command will return 9,25,49 and 81
print l[::2] # will return 1,9,25,49,81
print l[::-1] # will print backwards

############################

##### APPENDING TO A LIST AND A GENTLE INTRODUCTION TO USE OF %d 

suitcase = [] 
suitcase.append("sunglasses")

suitcase.append("tablet")
suitcase.append("camera")
suitcase.append("shorts")



list_length = len(suitcase)

print "There are %d items in the suitcase." % list_length ### %d refers to a decimal string operation. %s would be a character string 
print suitcase

############################

##### USING FOR WITH A LIST, AND OTHER COOL THINGS LIKE USING SORT()

# you can use FOR to loop through lists, strings and dictionary

start_list = [5, 3, 1, 2, 4]
square_list = []

for number in start_list:
    square_list.append(number**2)

square_list.sort()
print square_list

# to remove or delete items from a list see the example below

beatles = ["john","paul","george","ringo","stuart"]
beatles.remove("stuart")  ## you can also use del(beatles[4]) or beatles.pop(4) ... these methods of delete use index positions
print beatles

############################
'''
# General operations of LISTs


alist.append(item)	Adds a new item to the end of a list
alist.insert(i,item)	Inserts an item at the ith position in a list
alist.pop()	Removes and returns the last item in a list
alist.pop(i)	Removes and returns the ith item in a list
alist.sort()	Modifies a list to be sorted
alist.reverse()	Modifies a list to be in reverse order
del alist[i]	Deletes the item in the ith position
alist.index(item)	Returns the index of the first occurrence of item
alist.count(item)	Returns the number of occurrences of item
alist.remove(item)	Removes the first occurrence of item

'''


##### LIST AND FUNCTIONS

# an example that shows the use of a function in conjunction with concatenation 

n = "Hello"
def string_function(arg):
    return str(arg) + ' world'

print string_function(n)

# example of a function that returns average. Note the use of the float() function. 
# To divide two integers and end up with a float, you must first use float() to convert one of the integers to a float

def average(x):
    length = len(x)
    summary = sum(x)
    return float(summary)/length


	
# another example of list and functions in conjunction with modification an element in a list

n = [3, 5, 7]

def list_function(x):
    x[1] = x[1]+3 # we modify the second element here
    return x

print list_function(n)

# another example that combines the user input, function and appending to a list using .append(). It also introduces try and except.

n = [3, 5, 7]

def list_extender(x):
    append = raw_input("enter the number you want to add to the list")
    try:
        val = int(append) #int() to convert the string to an integer
        x.append(val)
        return x
    except ValueError:
        print("That's not an int!")    
        pass ###pass ignores errors and exceptions and allows the program to proceed. If you want the script to terminate use exit(). 
		
print list_extender(n)

#In the example below we split the text string using .split() and then again rejoin them using .join(). The example also illustrates 2 inputs to a function

def censor(text, word):
    text_list = text.split()
    for i in range(len(text_list)):
        if text_list[i] == word:
            text_list[i] = "*" * len(word)
        else: text_list[i] = text_list[i]
    joined_word = " ".join(text_list)
    
    return joined_word
    
print censor("a horse is happy", "happy")


# pass v/s exit v/s break. pass ignores errors and exceptions and allows program to proceed.
# exit() terminates a python script all together
# break exits from a loop

# Using Range. The Python range() function is just a shortcut for generating a list, so you can use ranges in all the same places you can use lists
#The range function has three different versions:
#range(stop)
#range(start, stop)
#range(start, stop, step)

n = range(0,3)

def my_function(x):
	net = 0
	#squares = [] #create a blank list
        for i in range(0, len(x)):
            x[i] = x[i] * 2
            net = net + x[i]
        return (x, net) #observe syntax for multiple outputs

print my_function(n) 

# another example of range() 

def reverse(text):
    count = len(text)
    print count
    word = text
    print word
    rev_word = ""
    for i in range(0, count):
        rev_word = rev_word + word[count-1]
        count = count - 1
    return rev_word    
print reverse("niharika")

#alternatively 

def reverse(text):
    count = len(text)
    print count
    word = text
    print word
    rev_word = ""
    for i in range(count-1, -1, -1):
        rev_word = rev_word + word[i]
        #count = count - 1
    return rev_word    
print reverse("niharika")

# an example that shows flattening of a list

n = [[1, 2, 3], [4, 5, 6, 7, 8, 9]]

def flatten(x):
    s = []
    for i in range(0,len(x)):
        s = s + x[i]
    
    return s

print flatten(n)

# another example that illustrates the use of "not in"

num_example = [1,1,2,2,3,3,4,4]

def remove_duplicates(nums):
    new_nums = []
    for i in range(len(nums)):
        if nums[i] not in new_nums:
            new_nums.append(nums[i])
    
    return new_nums
print remove_duplicates(num_example)

#### advanced list concept

# the following code creates a 2 dimensional list

board = []

for x in range(0, 5):
    board.append(["O"] * 5)
print board
def print_board(board):
    for row in board:
        print " ".join(row)

board[4][2] = "X" # this refers to the 3rd element on the 5th list

# some other ways to create a list. The technique below is called "list comprehension"

evens_to_50 = [i for i in range(51) if i % 2 == 0] # a list of all the even numbers from 0 to 50
even_squares = [i**2 for i in range(1,11) if i % 2 == 0] # a list of squares of integers between 1 and 10 that are even
threes_and_fives = [i for i in range(1,16) if i % 3 == 0 or i % 5 == 0]
all_caps = [ch.upper() for ch in 'comprehension' if ch not in 'aeiou']


line = '1234567890'
n = 2
split_by_twos =  [line[i:i+n] for i in range(0, len(line), n)] #creates ['12', '34', '56', '78', '90']

	
############################

##### DICTIONARIES

d = {} # creates an empty dictionary

d = {'key1' : 1, 'key2' : 2, 'key3' : 3} # This is a dictionary called d with three key-value pairs

dict_name[new_key] = new_value # adds a new key "new_key" to an existing dictionary "dict_name" and assigns value "new_value" to the new_key

# The length len() of a dictionary is the number of key-value pairs it has. Each pair counts only once, even if the value is a list

del dict_name[key_name] #Items can be removed from a dictionary with the del (delete) command

dict_name[existing_key] = new_value # A new value can be associated with a key by assigning a value to the key

# the use of [] in list v/s dictionary. list[1] returns 2nd element of the list. Dictionary[key_name] returns key value of that dictionary

# example of creating a dictionary with empty list

lloyd = {'name' : "Lloyd",
        'homework':[] ,
        'quizzes' : [],
        'tests' : [],
        }

# simple examples on dictionary
inventory = {
    'gold' : 500,
    'pouch' : ['flint', 'twine', 'gemstone'], # Assigned a new list to 'pouch' key
    'backpack' : ['xylophone','dagger', 'bedroll','bread loaf']
}

# Adding a key 'burlap bag' and assigning a list to it
inventory['burlap bag'] = ['apple', 'small ruby', 'three-toed sloth']

# Sorting the list found under the key 'pouch'
inventory['pouch'].sort() 

inventory['pocket'] = ['seashell', 'strange berry', 'lint'] # new key called pocket
inventory['backpack'].sort()
inventory['backpack'].remove('dagger') #removing an element from a list within a key
inventory['gold'] = 550 #updating numeric value of a key

print inventory

# Extracting information from a dictionary

#example 1

capitals = {'Iowa':'DesMoines','Wisconsin':'Madison', 'Utah': 'SaltLakeCity', 'California': 'Sacramento'}
for k in capitals:
   print(capitals[k]," is the capital of ", k)
   # print capitals[k] + " is the capital of " + k # this is a neater version

#example 2

my_dict = {
    "Student": ["ABJ","PBJ"],
    "Age": [25,28],
    "Course": ["data science","elec"]
}

print my_dict.items() # .items() function returns key/value pairs BUT not in any specific order
print my_dict.keys() # The keys() function returns an array of the dictionary's keys
print my_dict.values() # The values() function returns an array of the dictionary's values
print my_dict.get('Student', "Not there")	#Returns the value of key 'Student' i.e. the list ["ABJ","PBJ"]. If it cant find key 'Student', it will return "Not there".

### Use a for loop to go through the webster dictionary and print out all of the definition

webster = {
	"Aardvark" : "A star of a popular children's cartoon show.",
    "Baa" : "The sound a goat makes.",
    "Carpet": "Goes on the floor.",
    "Dab": "A small amount."
}

for keyvalue in webster:
    print webster[keyvalue]
	
### example showing how key and value can be extracted from a dictionary. The example also shows the use of str() because you cannot concatenate 'str' and 'float' objects
#In order for the 'for' statement to work here, you must have (1) some similar keys & (2) use the 'or' logic. If the keys in both dictionary are identical, then 'and' can be used
# example 1
prices = {'banana':4 , 'apple':2 , 'orange':1.5, 'pear':3}
stock =  {'banana':6 , 'apple':0 , 'orange':32, 'pear':15, 'grape':25}
for key in prices or stock: 
    print key + " price: "+ str(prices[key]) #note format. Use of key_name extracts name of the key. Use of dictionary_name[key_name] extracts key value
    
    print key + " stock: "+str(stock[key])
	
# example 2
prices = {
    "banana": 4,
    "apple": 2,
    "orange": 1.5,
    "pear": 3
}
    
stock = {
    "banana": 6,
    "orange": 32,
    "pear": 15,
    "apple": 0
}


total = 0
for key in prices and stock:
    value = prices[key] * stock[key]
    total = total + value
print total
   
# Example: Use of functions with dictionaries. Very cool example that shows (1) use of 'for' loop, (2) updates in key_values

groceries = {"banana":8, 
	"orange":2, 
	"apple":2
	}

stock = { "banana": 6,
    "apple": 0,
    "orange": 32,
    "pear": 15
}	
    
prices = { "banana": 4,
    "apple": 2,
    "orange": 1.5,
    "pear": 3
}

def compute_bill(food):
    something = 0
    for items in food:
		if stock[items] == 0:
			something = something + 0
		elif stock[items] >= groceries[items]:
			stock[items] = stock[items]-groceries[items]
			something = something + (prices[items] * groceries[items])
		else:
			something = something + (prices[items] * stock[items])
			stock[items] = 0
			
			

    return something    
      
print compute_bill(groceries)

# Example: Use of functions with dictionaries. Very cool example that shows (1) how it all comes together (list and dictionary) and (2) how one function calls another function.

students = [lloyd , alice , tyler]

lloyd = {
    "name": "Lloyd",
    "homework": [90, 97, 75, 92],
    "quizzes": [88, 40, 94],
    "tests": [75, 90]
}
alice = {
    "name": "Alice",
    "homework": [100, 92, 98, 100],
    "quizzes": [82, 83, 91],
    "tests": [89, 97]
}
tyler = {
    "name": "Tyler",
    "homework": [0, 87, 75, 22],
    "quizzes": [0, 75, 78],
    "tests": [100, 100]
}

def average(x):
    length = len(x)
    summary = sum(x)
    return float(summary)/length

def get_average(y): #observe how this function calls the top function
    hw = average(y["homework"])
    qz = average(y["quizzes"])
    ts = average(y["tests"])
    return (0.1*hw)+(0.3*qz)+(0.6*ts)
    
def get_letter_grade(score):
    score = round(score)
    if score >= 90: return "A"
    if score >= 80 and score < 90: return "B"
    if score >= 70 and score < 80: return "C"
    if score >= 60 and score < 70: return "D"
    else: return "F"
	
def get_class_average(student):
    class_score = 0
    for names in student:
        class_score = class_score + get_average(names)
        
    return float(class_score)/len(student)

print get_class_average(students)
print get_letter_grade(get_class_average(students)) #observe how 2 functions are nested here



#################### LOOPS

# While loop format

# example that also demonstrates format
loop_condition = True

while loop_condition:
    print "I am a loop"
    loop_condition = False

# example of while loop	in conjunction with else

from random import randrange

random_number = randrange(1, 10)
print random_number
count = 0
# Start your game!
while count < 4:
    guess = int(raw_input("Guess a number: "))
    if guess == random_number:
        print "you win"
        break
    else: count = count + 1
else: print "you lose"

# for loop example.	The example below add the sum of the individual integers in an integer. It also demos how a for loop can loop through a string

def digit_sum(x):
    str_x = str(x)
    count = 0
    for integer in str_x:
        y = int(integer)
        count = count + y
        
    return count
        
print digit_sum(1234567891011)

#################### LAMBDA

# the function the lambda creates is an anonymous function. i.e. you dont need to define it. 
# If a function will be used a lot, it makes sense to define it. when you need a quick function to do some work for you, lambda is useful

# lambda examples
my_list = range(16)
print filter(lambda x: x % 3 == 0, my_list) # notice the use of the filter() function. Here the function returns numbers between 0 and 15 that are divisible by 3 

squares = [x **2 for x in range(1,11)]
print filter(lambda i: i>=30 and i<=70, squares) # Here the function returns squares between 30 and 70

###############################################################
###															###
###															###
###   				Advanced Functions						###
###															###
###															###
###############################################################

#and introducing {0} and {1}

# You would use *args when you're not sure how many arguments might be passed to your function
def print_everything(*args):
        for count, thing in enumerate(args):
		print '{0}. {1}'.format(count, thing)

# the text below is output		
print_everything('apple', 'banana', 'cabbage')
0. apple
1. banana
2. cabbage

# **kwargs allows you to handle named arguments that you have not defined in advance
def table_things(**kwargs):
	for name, value in kwargs.items():
		print '{0} = {1}'.format(name, value)

# the text below is output
table_things(apple = 'fruit', cabbage = 'vegetable')
cabbage = vegetable
apple = fruit

# Create a helper function to call a function repeatedly
def repeat(times, func, *args, **kwargs):
    for _ in xrange(times):
        yield func(*args, **kwargs)

###############################################################
###															###
###															###
###   				Bitwise Operators						###
###															###
###															###
###############################################################

#The bin() function. bin() takes an integer as input and returns the binary representation of that integer in a string
print bin(1)
# You can also represent numbers in base 8 and base 16 using the oct() and hex() functions
print int("0b100",2) # int() 2nd parameter = 2, tells int() that the 1st parameter is binary

# Left Bit Shift (<<)  
0b000001 << 2 == 0b000100 (1 << 2 = 4)
0b000101 << 3 == 0b101000 (5 << 3 = 40)       

# Right Bit Shift (>>)
0b0010100 >> 3 == 0b000010 (20 >> 3 = 2)
0b0000010 >> 2 == 0b000000 (2 >> 2 = 0)

# The bitwise AND (&) operator compares two numbers on a bit level and returns a number where the bits of that number are turned on if the corresponding bits of both numbers are 1
     a:   00101010   42
     b:   00001111   15       
===================
 a & b:   00001010   10
# The bitwise OR (|) operator compares two numbers on a bit level and returns a number where the bits of that number are turned on if either of the corresponding bits of either number are 1 
    a:  00101010  42
    b:  00001111  15       
================
a | b:  00101111  47 

# The XOR (^) or exclusive or operator compares two numbers on a bit level and returns a number 
# where the bits of that number are turned on if either of the corresponding bits of the two numbers are 1, but not both

    a:  00101010   42
    b:  00001111   15       
================
a ^ b:  00100101   37

a = 0b11101110
mask = 0b11111111
print(bin(a^mask)) # the XOR (^) operator is very useful for flipping bits

# The bitwise NOT operator (~) just flips all of the bits in a single number

# A bit mask can help you turn specific bits on, turn others off, or just collect data from an integer about which bits are on or off

# example of a bit mask in a function. This function checks if a specific bit is on.

def check_bit4(integer):
    string = ""
    mask = 0b1000
    result = mask & integer
    if result == 0:
        string = "off"
    else: string = "on"
    
    return string
    
print check_bit4(0b0100)

# another example of bit max in a function. Here the function "flips" the nth bit of a number

def flip_bit(number, n):
    mask = (0b1 << n-1)
    result = number ^ mask
    return bin(result)

###############################################################
###															###
###															###
###   						CLASSES							###
###															###
###															###
###############################################################

# Example of class

class Fruit(object):
    """A class that makes various tasty fruits."""
    def __init__(self, name, color, flavor, poisonous):
        self.name = name
        self.color = color
        self.flavor = flavor
        self.poisonous = poisonous

    def description(self):
        print "I'm a %s %s and I taste %s." % (self.color, self.name, self.flavor) #use %d for integer, %f for float, %e for exponents less than −4 or greater than +5, otherwise use %f

    def is_edible(self):
        if not self.poisonous:
            print "Yep! I'm edible."
        else:
            print "Don't eat me! I am super poisonous."

lemon = Fruit("lemon", "yellow", "sour", False)

lemon.description()
lemon.is_edible()

'''

Additional formatting options
%20d	Put the value in a field width of 20
%-20d	Put the value in a field 20 characters wide, left-justified
%+20d	Put the value in a field 20 characters wide, right-justified
%020d	Put the value in a field 20 characters wide, fill in with leading zeros.
%20.2f	Put the value in a field 20 characters wide with 2 characters to the right of the decimal point.
%(name)d	Get the value from the supplied dictionary using name as the key

'''

# Example of class
 

class ShoppingCart(object):#Class shopping_cart inherits from the object class
    """Creates shopping cart objects
    for users of our fine website."""
    items_in_cart = {} #member variable available to all members of the class. Set to NULL in this example.
    def __init__(self, customer_name):# __init__() is the function that "boots up" each object the class creates. it contains atleast "self" argument.
        self.customer_name = customer_name

    def add_item(self, product, price): #add_item is a method for the class ShoppingCart 
        """Add product to the cart."""
        if not product in self.items_in_cart:
            self.items_in_cart[product] = price
            print product + " added."
        elif self.items_in_cart[product] != price:
            self.items_in_cart[product] = price
            print product + " is already in the cart. Price updated"
        else:
            print product + " is already in the cart."

    def remove_item(self, product):
        """Remove product from the cart."""
        if product in self.items_in_cart:
            del self.items_in_cart[product]
            print product + " removed."
        else:
            print product + " is not in the cart."

my_cart = ShoppingCart("aroon")

my_cart.add_item("computer",1000)
my_cart.add_item("computer",800)
my_cart.add_item("computer",800)
my_cart.add_item("zebra",10000)
print my_cart.customer_name
print my_cart.items_in_cart

# Example of class showing inheritance, overriding the method of base class and using super() to access method of the base class

class Employee(object): 
    """Models real-life employees!"""
    def __init__(self, employee_name):
        self.employee_name = employee_name

    def calculate_wage(self, hours):
        self.hours = hours
        return hours * 20.00

class PartTimeEmployee(Employee): # note the derived class (or subclass) does not need an __init__() function
    def calculate_wage(self, hours):# over here the derived class overrides calculate_wage method from base class
        self.hours = hours
        return hours * 12.00

    def full_time_wage(self, hours): #where as over here using the super() function, it accesses the method of base class
        self.hours = hours
        return super(PartTimeEmployee, self).calculate_wage(hours)
        
milton = PartTimeEmployee("milton")
print milton.full_time_wage(10)

# more examples showing syntax

class Triangle(object):
    number_of_sides = 3
    def __init__(self, angle1, angle2, angle3):
        self.angle1 = angle1
        self.angle2 = angle2
        self.angle3 = angle3
        
    def check_angles(self):
      if self.angle1+self.angle2+self.angle3==180:
        return True
      else:
        return False

class Equilateral(Triangle):
    angle = 60
    def __init__(self):
        self.angle1 = self.angle
        self.angle2 = self.angle
        self.angle3 = self.angle

        
my_triangle = Triangle(90,60,30)
print my_triangle.number_of_sides
print my_triangle.check_angles()

# More example showing the use of __repr__
#  __repr__() method is short for representation;
# by providing a return value in this method, we can tell Python how to represent an object of our class 

class Point3D(object):
    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z
    
    def __repr__(self):
        return "(%d, %d, %d)" % (self.x, self.y, self.z)
        
my_point = Point3D(1,2,3)

print my_point

# More examples for syntax and showing inheritance properties

class Car(object):
    condition = "new"
    def __init__(self, model, color, mpg):
        self.model = model
        self.color = color
        self.mpg   = mpg
        
    def display_car(self):
        print "This is a %s %s with %i MPG." %(self.color, self.model, self.mpg)
        
    def drive_car(self):
        self.condition = "used"
        print self.condition

class ElectricCar(Car):
    def __init__(self, model, color, mpg, battery_type): #although ElectricCar inherits from Car, we still specify all member variables
        self.model = model
        self.color = color
        self.mpg   = mpg
        self.battery_type = battery_type

my_car = ElectricCar("DeLorean", "silver", 88, "molten salt")
print my_car.condition
my_car.drive_car()

# More examples for syntax and showing __str__ , __add__ and __eq__ methods

class Fraction:
     def __init__(self,top,bottom):
         self.num = top
         self.den = bottom

     def __str__(self): #, __str__, is the method to convert an object into a string. Here we simply override this method with the name __str__ and give it a new implementation
         return str(self.num)+"/"+str(self.den)

     def show(self):
         print(self.num,"/",self.den)

     def __add__(self,otherfraction): #here we override the __add__ method.
         newnum = self.num*otherfraction.den + \
                      self.den*otherfraction.num
         newden = self.den * otherfraction.den
         common = gcd(newnum,newden)
         return Fraction(newnum//common,newden//common)

     def __eq__(self, other):# here we override the __eq__ method.(equality method)
         firstnum = self.num * other.den
         secondnum = other.num * self.den

         return firstnum == secondnum
		 
## more on class ... coordinate geometry

class Coordinate(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def getX(self):
        # Getter method for a Coordinate object's x coordinate.
        # Getter methods are better practice than just accessing an attribute directly
        return self.x

    def getY(self):
        # Getter method for a Coordinate object's y coordinate
        return self.y

    def __str__(self):
        return '<' + str(self.getX()) + ',' + str(self.getY()) + '>'

    def __eq__(self, other):
		# method that returns True if coordinates refer to same point in the plane (i.e., have the same x and y coordinate)
        # First make sure `other` is of the same type 
        assert type(other) == type(self)
        # Since `other` is the same type, test if coordinates are equal
        return self.getX() == other.getX() and self.getY() == other.getY()

    def __repr__(self):
        return 'Coordinate(' + str(self.getX()) + ', ' + str(self.getY()) + ')'

#################### PYTHON FILE I/O 

# Example 

my_list = [i**2 for i in range(1,11)]

f = open("output.txt", "w") #open output.txt in "w" mode ("write") and store the result of this operation in a file object, f
							# use "r+" as a second argument to the function so the file will allow you to read AND write to it!
							# use "r" as a second argument to the function so the file will allow you to read it!

for item in my_list:
    f.write(str(item) + "\n") #"\n" (newline) to ensure each will appear on its own line

# print my_file.read() #.read() function reads my_file	
f.close()  #Always close the connection

# Example of readline()

my_file = open("text.txt", "r")

print my_file.readline() #reads 1st line
print my_file.readline() # reads second line
print my_file.readline() # reads 3rd line

my_file.close()

import csv
inputlist = [[(3, 'c'), ('Python', 'CWI')], [(1, 'a'), ('Spark', 'Berkeley')], [(4, 'd'), ('Hive', 'Apache')]]
out = csv.writer(open("myfile.csv","w"), delimiter=',',quoting=csv.QUOTE_NONE)
out.writerows(inputlist) #writerows creates rows vs adding all in one row


#################### MISCELLANEOUS COOL TRICKS

#random
import random
random.random()        # Random float x, 0.0 <= x < 1.0
0.37444887175646646
random.uniform(1, 10)  # Random float x, 1.0 <= x < 10.0
1.1800146073117523
random.randint(1, 10)  # Integer from 1 to 10, endpoints included
7
random.randrange(0, 101, 2)  # Even integer from 0 to 100
26
random.choice('abcdefghij')  # Choose a random element
'c'

items = [1, 2, 3, 4, 5, 6, 7]
random.shuffle(items)
items
[7, 3, 2, 5, 6, 4, 1]

random.sample([1, 2, 3, 4, 5],  3)  # Choose 3 elements
[4, 1, 5]

#use of .join() method
board = [] # create an empty list

for x in range(0, 5):
    board.append(["O"] * 5) #append 0 to the list and loop this 5 times to create 5 lists

def print_board(board):
    stra = " "
    for row in board:
        print stra.join(row) #The method join() returns a string in which the string elements of sequence have been joined by stra separator
		
# another example of .join		

animals = ['cat', 'dog','rabbit']
animalstring = ''.join(animals) #using .join list is converted into a string
print animalstring
list1 = [x for x in animalstring] # every element of string is converted into a list
print list1
string1 = ''.join(sorted(set(animalstring), key=animalstring.index)) # duplicate characters are removed from the string and order is maintained. sorted() function on display.
print string1
print type(string1)

# the type() function returns the type of variable		
type(variable_name)

#math
import math
math.sqrt(16)
math.pi #returns the value of pi

import os.path
baseDir = os.path.join('data')
inputPath = os.path.join('cs100', 'lab1', 'shakespeare.txt')
fileName = os.path.join(baseDir, inputPath)
print fileName # returns 'data/cs100/lab1/shakespeare.txt'

#################### SOME VERY COOL AND SIMPLE CODE

# finding greatest common divider

def gcd(m,n):
    while m%n != 0:
        oldm = m
        oldn = n

        m = oldn
        n = oldm%oldn
    return n
	

###############################################################
###															###
###															###
###   				Regular expressions						###
###															###
###															###
###############################################################

import re

# go to http://pythex.org/   or https://regex101.com/

print (re.split('\s', "here are some words that are fantastic")) # re.split looks for the regular expression, in this case space (\s). Returns a list.


print (re.split(r'[a-f]', "here are some words that are fantastic"))  # [a-f] signifies to look for range of lower characters a through f. Note [^5] will match any character except '5'. the caret '^' when included at the start of [ suggest match all except.

print (re.split(r'[a-f][r]', "here are some words that are fantastic"))  # [a-f][r] looks for combinations of ar, br, cr, ... fr

print (re.findall(r'\d', "my address is 405 S Lynnwood trail")) #\d looks for digits ... will return a list of '4', '0', '5'

print (re.findall(r'\d{1,5}', "my address is 405 S Lynnwood trail")) # {1,5} looks for a continuous range (from 1 to 5 digits) ... will return 405

print (re.findall(r'\d{1,5}\s\w*', "my address is 405 South Lynnwood trail")) # \s\w looks for a space followed by alphanumeric characters. The * suggests to look for as many alphanumeric ... will return 405 South

print (re.findall(r'\d{1,5}\s\w*.*\.', "my address is 405 South Lynnwood trail.gobbledigool")) # here after w*, we ask python to look for all characters denoted by .* ... the dot '.' itself matches only character. But when combined with * will look for everything after w*. However we want to end the search at a literal '.' ... therefore we use \. indicating that we end the search with a literal .    ................... this code will return 405 South Lynnwood trail.


print re.findall(r'^\S+', "my address is 405 South Lynnwood trail.gobbledigool") # ^ Matches the start of the string. Here ^\S+ looks for all non-white spaces at the start. Returns [my]

# sophisticated example of regular expression

APACHE_ACCESS_LOG_PATTERN = '^(\S+) (\S+) (\S+) \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+)\s*(\S*)" (\d{3}) (\S+)'


logline = '127.0.0.1 - - [01/Aug/1995:00:00:01 -0400] "GET /images/launch-logo.gif HTTP/1.0" 200 1839'

match = re.search(APACHE_ACCESS_LOG_PATTERN, logline)

print match.group(1) # the (...) helps match regular expressions in group to facilitate easy extraction of data


print re.findall('[([\w:/]+\s[+\-]\d{4})\]' ,'- - [01/Aug/1995:00:00:01 -0400')




###############################################################
###															###
###															###
###   				Creating fake data						###
###															###
###															###
###############################################################

from faker import Factory
fake = Factory.create()
fake.seed(4321)

print dir(fake) # gives you all the attributes of fake
# for example:
for _ in range(0,10):
  print fake.military_state()
 
for _ in range(0,10):
  print fake.name()

# creating lots of fake data
email = []
for _ in range(0,10):
  email.append(fake.email())

datetime = []
for _ in range(0,10):
  datetime.append(fake.date_time())

address = []
for _ in range(0,10):
  address.append(fake.address())
 
city = []
for _ in range(0,10):
  city.append(fake.city())
  
state = []
for _ in range(0,10):
  state.append(fake.state())
  
mylist = [list(a) for a in zip(email, datetime, address, city, state)]




