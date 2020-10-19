#!/usr/bin/env python3


import numpy as np
from numpy.linalg import solve, inv
from typing import Tuple, List





class Simplex:
	''' Basic implementation of the 1 phase simplex method for
		linear programming problem '''
		
		
	def __init__(self, A: List[List[float]], b: List[float], c: List[float]):
		
		self.A = np.array(A)
		self.b = np.array(b)
		self.c = np.array(c)
		
		self.m = self.A.shape[0]					# number of constraint
		self.n = self.A.shape[1] 					# number of variables
		self.p = self.n - self.m					# number of non basic variables
		
		self.itermax = 100
		


	def computeBasicVariables(self) -> np.array:
		''' Solves a linear system '''
		
		try:
			
			self.xb = solve(self.B, self.b)
			
			return self.xb
		
		except	LinAlgError:
			
			print('\n\tOh no, basic matrix is singular!!')



	def reducedCost(self) -> np.array:
		''' Computes the reduced cost associated to the current basis '''
	
		return self.cn - np.dot(self.cb, np.dot(inv(self.B), self.N))
		
	

	def optimalityTest(self, rcost: np.array, Bland: bool = True) -> Tuple[bool, int]:				# the problem is meant a minimization
		''' Checks for optimality and returns the failed indexes '''
			
		h = self.n + 1													# initialize this two values
		optimal = True
		
		for i, indx in enumerate(self.index_n):
			
			if rcost[i] < 0:											# set of indexes failing the optimality test
				
				optimal = False
				
				if indx < h:	h = indx								# Bland's rule
		
		return optimal, h
	
	
	def Unbounded(self, h: int, Bland: bool = True) -> Tuple[bool, float, int]:
		''' Finds the index to be pulled out of the basis:
			indx:		index in which the optimality test fails (non basic index)'''
				
		Binv = inv(self.B)
			
		num = Binv.dot(self.b)
		den = Binv.dot(self.A[:, h])
		
		unbounded = False
		k = self.n * 2
		theta = 10 ** 10
		
		for i, indx in enumerate(self.index_b):
			
			div = num[i] / den[i]
			
			if div < 0:													# the problem is unbounded
				
				return True, -1 , -1
			
			if div < theta or (div == theta and indx < k):				# Bland's rule
				
				theta = div
				k = indx
			
		return False, theta, k
		


	def newBasis(self, k: int, h: int, theta) -> Tuple[list, list]:
		''' Returns and updates the indexes of the new basis '''

		new_b = [h if i == k else i for i in self.index_b]				# updated indices
		new_n = [k if i == h else i for i in self.index_n]
		
		for i, indx in enumerate(new_b):								# updates the basic solution
			
			if indx == k:
				
				self.xb[i] = theta
			
				break
		
		self.index_b = new_b
		self.index_n = new_n
	
		self.update()													# computes the new partitions
		
		return new_b, new_n
	
	
	def update(self) -> None:
		''' Updates the A and c partitions '''
		
		self.B = self.A[:, self.index_b]
		self.N = self.A[:, self.index_n]
		
		self.cb = self.c[self.index_b]
		self.cn = self.c[self.index_n]
	
	
	def assembleSolution(self):
		''' Assembles the full state vector x to be returned '''
		
		x = [0 for i in range(self.n)]
		
		for index, value in zip(self.index_b, self.xb):
			
			x[index] = value
		
		return x
		
		
	
	def algorithm(self, index_b_guess: List[int]) -> List[float]:
		''' Single iteration of the algorithm '''
		
		assert len(index_b_guess) == self.m,	'\n\tBasis index vector is out of range!!'
		
		self.index_b = np.array(index_b_guess)
		self.index_n = [i for i in range(self.n) if i not in self.index_b]		# remaining indexes
		
		self.update()
		
		for i in range(self.itermax):
		
			self.computeBasicVariables()
			
			reduced_cost = self.reducedCost()
			
			optimal, h = self.optimalityTest(reduced_cost)
			
			if optimal:
				
				if len(self.xb) < 20:
				
					print('\n\tThe variable: x = {} is optimal'.format(self.assembleSolution()))
				
				return list(self.xb)
			
			unbouded, theta, k = self.Unbounded(h)
			
			if unbouded:
				
				print('\n\tThe current problem is unbounded!!')
				
				return []
				
			self.newBasis(k, h, theta)
	
	
