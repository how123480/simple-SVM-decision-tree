import numpy as np
from qpsolvers import solve_qp

P = np.array([[1e-12,0,0],\
			[0,1,0],\
			[0,0,1]],dtype=float)
q = np.zeros(3)

X = np.array([[1.,4.,3.], [1.,7.,2.], [1.,4.,8.], [1.,2.,1.], [1.,2.,-1.],[1.,-1.,3.], [1.,-1.,-2.]], dtype=float)
Y = np.array([1,1,1,-1,-1,-1,-1], dtype=float)
#X = -1 * np.array([x*y for x,y in zip(X,Y)], dtype=float)

G = np.array([[-1., -4., -3.], [-1., -7., -2.], [-1., -4., -8.], [1., 2., 1.], [1., 2., -1.], [1., -1., 3.], [1., -1., -2.]])
h = -1 * np.ones(7).reshape((7,))
print("P: ", P) 
print("q: ", q)
print("G: ", G)
print("h: ", h)
solve = solve_qp(P, q, G, h)
weight = solve[1:]
bias = solve[0]

print("weight:", weight)
print("b: ", bias)
# find support vector
for i in range(X.shape[0]):
	f = np.dot(weight.T, X[i][1:]) + bias
	print("i: {}, f(x_i): {}".format(i, f)) 