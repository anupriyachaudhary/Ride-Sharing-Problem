import numpy
import random
import pylab
import math
import matplotlib.pyplot as plt
from itertools import combinations
import networkx as nx
from networkx.algorithms import bipartite
from scipy.spatial import distance
import time

# number of requests and numbers of drivers
n = 50
m = 2*n

############################################
############ GENERATING INPUTS #############
############################################

# Distribution variables
B = 100
no_of_centers = 20
sigma = 20

## Generate inputs from uniform distribution
s = ((numpy.random.rand(m,2))*B).tolist()
t = ((numpy.random.rand(m,2))*B).tolist()
d = ((numpy.random.rand(n,2))*B).tolist()

##Gaussian distribution, N (mu, sigma^2)
#mu = ((numpy.random.rand(no_of_centers,2))*B).tolist()
#s = []
#t = []
#d = []
#for i in range(0, m):
#    s.append(B*sigma * (numpy.random.randn(1, 2)) + mu[random.randint(0,no_of_centers-1)])
#for i in range(0, m):
#    t.append(B*sigma * (numpy.random.randn(1, 2)) + mu[random.randint(0,no_of_centers-1)])
#for i in range(0, n):
#    d.append(B*sigma * (numpy.random.randn(1, 2)) + mu[random.randint(0,no_of_centers-1)])

############################################
############ NYC DATASET INPUT##############
############################################

#EarthRadius = 6371000
#
#def positionRepresentation(longitude,latitude):
#    x = EarthRadius * math.cos(latitude) * math.cos(longitude)
#    y = EarthRadius * math.cos(latitude) * math.sin(longitude)
#    z = EarthRadius * math.sin(latitude)
#    return [x,y,z]
#
#def earthDistance(pos1, pos2):
#    return math.sqrt( math.pow((pos2[0] - pos1[0]),2) + math.pow((pos2[1]-pos1[1]),2) + math.pow((pos2[2]-pos1[2]),2))
#
#datafile = open('nycData_200_400_jfk.csv','r')
#data = numpy.matrix(numpy.genfromtxt('nycData_200_400_jfk.csv',delimiter=','))
#datafile.close()
#
#driverfile = open('nycDrivers_200_400_jfk.csv','r')
#drivers = numpy.matrix(numpy.genfromtxt('nycDrivers_200_400_jfk.csv',delimiter=','))
#driverfile.close()
#
#s = []
#t = []
#d = []
#
#sources = numpy.hstack((data[:,0],data[:,1]))
#destinations = numpy.hstack((data[:,3],data[:,4]))
#
#for i in range(len(sources)):
#    s.append(positionRepresentation(sources[i,0],sources[i,1]))
#for i in range(len(destinations)):
#    t.append(positionRepresentation(destinations[i,0],destinations[i,1]))
#for i in range(len(drivers)):
#    d.append(positionRepresentation(drivers[i,0],drivers[i,1]))

    
############################################
# GREEDY ALGORITHM FOR RIDE-SHARING (m = 2n)
############################################

# Calculate maximum of u(ij) & u(ji) and corresponding first & second pick-up
def min_max_dist(s, t, i, j):
    dist = (min(distance.euclidean(s[i], s[j]) + distance.euclidean(s[j], t[i]) + distance.euclidean(t[i], t[j]),
                distance.euclidean(s[i], s[j]) + distance.euclidean(s[j], t[j]) + distance.euclidean(t[j], t[i])),
            min(distance.euclidean(s[j], s[i]) + distance.euclidean(s[i], t[i]) + distance.euclidean(t[i], t[j]),
                distance.euclidean(s[j], s[i]) + distance.euclidean(s[i], t[j]) + distance.euclidean(t[j], t[i])))
    max_dist = max(dist)
    min_index = dist.index(max_dist)
    ret = []
    if min_index == 0:
        ret = [i, j, max_dist*(-1)]
    else:
        ret = [j, i, max_dist*(-1)]
    return ret

# Calculate the shortest route for pick up of i & j and corresponding first pick-up & second pick-up
def min_dist(s, t, i, j):
    dist = (distance.euclidean(s[i], s[j]) + distance.euclidean(s[j], t[i]) + distance.euclidean(t[i], t[j]),
            distance.euclidean(s[i], s[j]) + distance.euclidean(s[j], t[j]) + distance.euclidean(t[j], t[i]),
            distance.euclidean(s[j], s[i]) + distance.euclidean(s[i], t[i]) + distance.euclidean(t[i], t[j]),
            distance.euclidean(s[j], s[i]) + distance.euclidean(s[i], t[j]) + distance.euclidean(t[j], t[i]))
    min_dist = min(dist)
    min_index = dist.index(min_dist)
    ret = []
    if min_index == 0 or min_dist == 1:
        ret = [i, j, min_dist*(-1)]
    else:
        ret = [j, i, min_dist*(-1)]
    return ret

start_time = time.time() 
# make graph G1, with 2n requests as nodes and edges consisting of 2 nodes with weight = shortest 
# distance  to serve request pair but the worst one
G1 = nx.Graph()
G1.add_nodes_from(range(0, m))
for i in range(0,m):
    for j in range(i+1,m):
        G1.add_edge(i, j, weight = min_max_dist(s, t, i, j)[2])
# generate a minimum weight matching among riders/requests
matching1 = nx.max_weight_matching(G1, maxcardinality=True)

matched_riders = [] 
matched_riders_wt = []
for x in matching1:
    pairs = min_max_dist(s, t, x[0], x[1])
    matched_riders.append((pairs[0], pairs[1]))
    matched_riders_wt.append(pairs[2])
    
# make graph G2, with n requests pairs & n drivers as nodes 
# calculate minimum weight perfect matching  
G2 = nx.Graph()
for i in range(0,n):
    G2.add_node(matched_riders[i], bipartite=0)
    G2.add_node(i, bipartite=1)
   
for i in range(0,n):
    for j in range(0,n):
        first_rider = matched_riders[i][0]
        second_rider =  matched_riders[i][1]
        wt = (-1)*min(distance.euclidean(d[j], s[first_rider]), distance.euclidean(d[j], s[second_rider]))
        G2.add_edge(matched_riders[i], j, weight = wt)
# generate a minimum weight matching among riders/requests
matching2 = nx.max_weight_matching(G2, maxcardinality=True)

matched_wt = []
for x in matching2:
    matched_wt.append(G2[x[0]][x[1]]['weight'])

cost_M = (sum(matched_riders_wt) + sum(matched_wt))*(-1)

print("--- %s seconds ---" % (time.time() - start_time))

#############################################
########## LOWER BOUND ON OPTIMAL############
#############################################
    
# make graph G1, with 2n requests as nodes and edges consisting of 2 nodes with weight = shortest 
# distance  to serve request pair
G1_min = nx.Graph()
G1_min.add_nodes_from(range(0, m))
for i in range(0,m):
    for j in range(i+1,m):
        G1_min.add_edge(i, j, weight = min_dist(s, t, i, j)[2])
# generate a minimum weight matching among riders/requests
matching1_min = nx.max_weight_matching(G1_min, maxcardinality=True)

matched_riders_wt = []
for x in matching1_min:
    matched_riders_wt.append(G1_min[x[0]][x[1]]['weight'])

# make graph G2, with n requests pairs & n drivers as nodes     
G2_min = nx.Graph()
for i in range(0,m):
    G2_min.add_node(i,bipartite=0)

for j in range(m,n+m):
    G2_min.add_node(j,bipartite=1)
    
for i in range(0,m):
    for j in range(m,n+m):
        wt = (distance.euclidean(d[j-m], s[i]))*(-1)
        G2_min.add_edge(i, j, weight = wt)
# generate a minimum weight matching among riders/requests
matching2_min = nx.max_weight_matching(G2_min, maxcardinality=True)

matched_wt = []
for x in matching2_min:
    matched_wt.append(G2_min[x[0]][x[1]]['weight'])
L = (sum(matched_wt)+ sum(matched_riders_wt))*(-1)

print(cost_M/L)



#G1_t = nx.Graph()
#G1_t.add_nodes_from(range(0, m))
#for x in matching1:
#    G1_t.add_edge(x[0], x[1])
#nx.draw(G1_t)
    


   





