class Vector :
	
	def __init__ (self, coordinates):
		try:
			if not coordinates:
				raise ValueError
			self.coordinates = list(coordinates)
			self.dimension = len(coordinates)
		
		except ValueError:
			raise ValueError ('The coordinates must be nonempty')
		except TypeError: 
			raise TypeError (' The coordiantes must be iterable')
	
	def __str__ (self):
		return 'Vector: {}'.format(self.coordinates)
		
	def __eq__ (self, v):
		return self.coordinates == v.coordinates
	
	#Now I am confused, why would I need self in the above methods,
	#but not here?
	def add (firstV, secondV):
		if firstV.dimension != secondV.dimension:
			return 
		
		newVector = []
		for x in range(firstV.dimension):
			newVector[x] = (firstV.coordinates[x] + secondV.coordinates[x])
		
		return newVector
		
vectorOne = Vector.add(Vector(1, 2, 3), Vector(1, 2, 3))
