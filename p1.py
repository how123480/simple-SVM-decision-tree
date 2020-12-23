import numpy as np

features_map = [
	{0:"F",1:"M"},
	{0:"Family", 1:"Sport", 2:"Luxury"},
	{0:"Samll", 1:"Medium",2:"Large",3:"Extra Large"}
]
feature_name = {
	0:"Gender",
	1:"Car Type",
	2: "Shirt Size"
}
split_map = [
	[
		[[0],[1]]
	],
	[
		[[0],[1,2]],
		[[1],[0,2]],
		[[2],[0,1]]
	],
	[
		[[0],[1,2,3]],
		[[1],[0,2,3]],
		[[2],[0,1,3]],
		[[3],[0,1,2]],
		[[0,1],[2,3]],
		[[0,2],[1,3]],
		[[0,3],[1,2]]
	]
]

def _gini(p):
	N = len(p)
	l_0 = sum([1 for i in range(N) if(p[i][3] == 0)]) # number of class 0
	l_1 = N - l_0
	return 1- (l_0/N)**2 - (l_1/N)**2

def split(t, p, li, ri):
	N = len(p)
	left = np.array([p[i] for i in range(len(p)) if( p[i][t] in li)])
	right = np.array([p[i] for i in range(len(p)) if (p[i][t] in ri)])

	n_l = len(left)
	n_r = len(right)

	if(n_l == 0 or n_r == 0):
		return left, right, 1000

	g_l = _gini(left)
	g_r = _gini(right)

	return left, right, (n_l* g_l + n_r * g_r )/N

def build_tree(layer, p, tag):
	left, right = None, None
	min_gini = 1

	if(len(p) == 1):
		return 

	same_class = True
	same_f = True
	for i in range(len(p)-1):
		if(p[i][3] != p[i+1][3]):
			same_class = False
		if(sum(p[i][:3] == p[i+1][:3] ) != len(p[i][:3])):
			same_f = False
	# if all data belong to same class or have same features
	if(same_class or same_f):
		return 

	print("layer: {}, {}".format(layer,tag))
	split_num = [1,3,7]
	for f in range(3):
		for s in range(split_num[f]):

			l,r, g = split(f, p, split_map[f][s][0],split_map[f][s][1])

			if(g == 1000):
				# can't split (this features are the same)
				continue

			print("\tSplit with {} ({}, {}), gini: {}".format(
				feature_name[f],
				[features_map[f][i] for i in split_map[f][s][0]],
				[features_map[f][i] for i in split_map[f][s][1]],
				g))

			if(g < min_gini): 
				li,ri = split_map[f][s][0],split_map[f][s][1]
				t = f 
				left = l
				right = r	
				min_gini = g

	
	if(left is None or right is None):
		# can't split (all features are the same)
		print("All features are same, can't split!")
		return 
	
	print("features: ", feature_name[t])
	print("li:", [features_map[t][i] for i in li])
	print("ri: ",[features_map[t][i] for i in ri])
	print("left:", left[:,4])
	print("right: ", right[:,4])
	print("gini: ",min_gini)
	print("-"*100)

	build_tree(layer+1,left,"left_node")
	build_tree(layer+1,right,"right_node")


data = [
	[1,1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0],
	[0,1,1,1,1,1,1,1,1,2,0,0,0,2,2,2,2,2,2,2],
	[0,1,1,2,3,3,0,0,1,2,2,3,1,3,0,0,1,1,1,2],
	[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
	[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
]
data = np.array(data).T
print("data:\n{}".format(data))
print("gini: ", _gini(data))
print("-"*100)
build_tree(0,data,"root")
#print(data)
