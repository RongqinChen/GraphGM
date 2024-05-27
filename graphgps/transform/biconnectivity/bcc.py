# adapted from https://www.geeksforgeeks.org/biconnected-components/

# Python program to find biconnected components in a given undirected graph
# Complexity : O(V + E)
from typing import Tuple, List
from collections import defaultdict

# This class represents an directed graph using adjacency list representation
class Graph4BCC:

	def __init__(self, edge_list: List[Tuple[int, int]]):
		
		# default dictionary to store graph
		self.graph = defaultdict(set)
		for edge in edge_list:
			self.graph[edge[0]].add(edge[1]) 
			self.graph[edge[1]].add(edge[0])

		# No. of vertices
		self.V = len(self.graph)

		# time is used to find discovery times
		self.time = 0
		
		# Count is number of biconnected components
		self.count = 0
		self.bcc_list = []

	'''A recursive function that finds biconnected components using DFS traversal
	u --> The vertex to be visited next
	disc[] --> Stores discovery times of visited vertices
	low[] -- >> earliest visited vertex (the vertex with minimum
			discovery time) that can be reached from subtree
			rooted with current vertex
	st -- >> To store visited edges'''
	def BCCUtil(self, u, parent, low, disc, st):

		# Count of children in current node 
		children = 0

		# Initialize discovery time and low value
		disc[u] = self.time
		low[u] = self.time
		self.time += 1

		# Recur for all the vertices adjacent to this vertex
		for v in self.graph[u]:
			# If v is not visited yet, then make it a child of u
			# in DFS tree and recur for it
			if disc[v] == -1 :
				parent[v] = u
				children += 1
				st.append((u, v)) # store the edge in stack
				self.BCCUtil(v, parent, low, disc, st)

				# Check if the subtree rooted with v has a connection to
				# one of the ancestors of u
				# Case 1 -- per Strongly Connected Components Article
				low[u] = min(low[u], low[v])

				# If u is an articulation point, pop 
				# all edges from stack till (u, v)
				if parent[u] == -1 and children > 1 or parent[u] != -1 and low[v] >= disc[u]:
					self.count += 1 # increment count
					w = -1
					bcc = []
					self.bcc_list.append(bcc)
					while w != (u, v):
						w = st.pop()
						bcc.append(w)
					# 	print(w, end=" ")
					# print()
			
			elif v != parent[u] and low[u] > disc[v]:
				'''Update low value of 'u' only of 'v' is still in stack
				(i.e. it's a back edge, not cross edge).
				Case 2 
				-- per Strongly Connected Components Article'''

				low[u] = min(low [u], disc[v])
	
				st.append((u, v))


	# The function to do DFS traversal. 
	# It uses recursive BCCUtil()
	def BCC(self):
		
		# Initialize disc and low, and parent arrays
		disc = [-1] * (self.V)
		low = [-1] * (self.V)
		parent = [-1] * (self.V)
		st = []

		# Call the recursive helper function to 
		# find articulation points
		# in DFS tree rooted with vertex 'i'
		for i in range(self.V):
			if disc[i] == -1:
				self.BCCUtil(i, parent, low, disc, st)

			# If stack is not empty, pop all edges from stack
			if st:
				self.count = self.count + 1
				bcc = []
				self.bcc_list.append(bcc)
				while st:
					w = st.pop()
					bcc.append(w)
				# 	print(w,end=" ")
				# print ()

		return self.bcc_list
		

if __name__ == "__main__":
# Create a graph given in the above diagram

	edge_list = [
		(0, 1), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (1, 5), (5, 1),
		(0, 6), (5, 6), (5, 7), (5, 8), (7, 8), (8, 9), (10, 11)
	]
	g = Graph4BCC(edge_list)
	bcc_list = g.BCC()
	for bcc in bcc_list:
		print(bcc)

	print ("Above are % d biconnected components in graph" %(g.count));

# This code is contributed by Neelam Yadav
