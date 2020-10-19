#!/usr/bin/env python3

import numpy as np
from Simplex import Simplex


if __name__ == '__main__':
	
	A = [[3, 1, -1, 0],
		 [1, 1, 0, -1]]
	
	b = [3, 2]
	
	c = [3, 2, 0, 0]
	
	index_b_guess = [0, 2]		# initial guess of the basis index
	
	
	smplx = Simplex(A, b, c)
	
	smplx.algorithm(index_b_guess)
