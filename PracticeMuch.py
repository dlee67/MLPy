to_ten = [1, 1, 1, 4, 5, 6, 7, 8, 9, 10]
counter = 0

while counter != len(to_ten):
	if to_ten[counter] == 5:
		counter += 1
		continue	
	print(to_ten[counter])
	if to_ten[counter] == 7:
		break
	counter += 1
	