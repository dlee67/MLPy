from stuff import sick, cool

print(sick.sayHi())
print(cool.sayBi())

#Python has something called while-else
#Pretty fancy
to_ten = [1, 1, 1, 4, 6, 7, 8, 9, 10]
counter = 0

while counter != len(to_ten):
	print("Got: ", to_ten[counter]) 
	if to_ten[counter] == 5:
		print("Got: ", to_ten[counter], " Breaking out of the loop.")
		break	
	counter += 1
else:
	print("Didn't got five :(")
	
# Python's implementation with for-loop is pythonic
print("I am gonna do my for loop now")
# Conveniently enough, the iterable objects in python is
# lot more comfortable to deal with than C's
for num in to_ten:
	print(num)