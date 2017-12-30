# The values I've generated are returned
def generate():
	for i in range(10):
		yield i + 2
		
# here.		
for i in generate():
	print(i)
	
# yield keyword is specifically for generators,
# where yield only returns the generator object.
# 
# https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
# The user e-satis explains this concept well.
