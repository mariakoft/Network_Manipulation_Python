# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 01:59:34 2018

@author: Koft
"""

import pandas as pd
import numpy as np
import scipy as sc
from scipy import stats
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xlsxwriter
import os

#sm.xls
#Ask file name from user and import it
filename = input ("Pease add your xls or xlsx file's  filename: ");
xl = pd.ExcelFile(filename)
name=os.path.splitext(filename)[0]  #extract the name of the file without the type
head=input ("Does the file contain headers? Answser Yes or No: ");

if len(head)== 3: 
    df1 = xl.parse('Sheet1') # Load a sheet into a DataFrame 
else:
    df1 = xl.parse('Sheet1',header=None) 
    
#Check if labels are asigned to each node (row)
print ("Weight matrix size :" ) 
print(df1.shape)
dim=input ("Is the matrix NxN ? Anwser Yes or No: ");
if len(dim)==2:     
    df2=df1[df1.columns[0]]
    labels_dict=df2.to_dict()   #asign the labels from 1st column to nodes
    df1=df1.drop(df1.columns[[0]], axis=1) #delete first column whether requested
    weight_mat=pd.DataFrame.as_matrix(df1) #create weight matrix
    H = nx.from_numpy_matrix(np.array(weight_mat))
    G = H.to_directed(H) #create the graph
    nx.set_node_attributes(G, labels_dict, 'label')
    nx.draw(G, labels=labels_dict, with_labels=True)    #and visualize it
    plt.savefig(name + "_GraphPlot.png", format="PNG")
else:
    df1=df1
    labels_dict={i: i for i in range(df1.shape[1])}            #Create Node Labels
    #Create graph 1 and the adjacency matrix and plot it
    weight_mat=pd.DataFrame.as_matrix(df1) #create weight matrix
    H = nx.from_numpy_matrix(np.array(weight_mat))
    G = H.to_directed(H) #create the graph
    nx.draw(G,with_labels=True)    #and visualize it
    plt.savefig( name +"_GraphPlot.png", format="PNG")
    
    
    
#G=nx.path_graph(4)
nx.write_pajek(G, name + "_Graph"+ ".net")
pos = nx.spring_layout(G)  #for centrality plots used latter
print(nx.info(G)) 	#get network's info

#create an excel file to save my results

workbook  = xlsxwriter.Workbook('Reults_' + name + '.xlsx')
worksheet1 = workbook.add_worksheet('Adjacency Matrix')
worksheet2 = workbook.add_worksheet('Node Attributes')
worksheet3 = workbook.add_worksheet('Network Metrics')

#Choose format types for specify names
cell_format1 = workbook.add_format()
cell_format1.set_bold()
cell_format1.set_font_color('000080')
cell_format1.set_font_name('Times New Roman')
cell_format1.set_font_size(10)
cell_format1.set_indent(1)

cell_format2 = workbook.add_format()

#Calculate Adjacency Matrix and add it to workbook
adj_mat=nx.to_numpy_matrix(G)
A=np.asarray(adj_mat)

row = 0

for col, data in enumerate(A):
    worksheet1.write_column(row, col, data)

#Basic network Metrices

#Density
density = nx.density(G)
print("Network density:", density)

#Transitivity
triadic_closure = nx.transitivity(G)
print("Triadic closure:", triadic_closure)

#Average shortest path length 
Path_Lenght_G=nx.average_shortest_path_length(G, weight='weight')


#Calculate Clustering Coefficience
Clust_G=nx.average_clustering(H)   #implemented only for undirected graphs

############  Centralities ##################
#in-Degree Centrality
indegree_dict = nx.in_degree_centrality(G) 
nx.set_node_attributes(G, indegree_dict, 'in-degree')
print("Degree centrality:", indegree_dict)
	
#out-Degree Centrality
outdegree_dict = nx.out_degree_centrality(G) 
nx.set_node_attributes(G, outdegree_dict, 'out-degree')
print("Degree centrality:", outdegree_dict)

#Betweenness centrality
betweenness_dict = nx.betweenness_centrality(G, weight= 'weight') 
nx.set_node_attributes(G, betweenness_dict, 'betweenness')	# Assign each to an attribute in your network
print("Betweenness centrality:", betweenness_dict)

#in-Closeness Centrality
incloseness_dict = nx.closeness_centrality(G)
nx.set_node_attributes(G, incloseness_dict, 'in-closeness')
print("in-Closeness centrality:", incloseness_dict)

#out-Closeness Centrality
R=G.reverse()
outcloseness_dict = nx.closeness_centrality(R)
nx.set_node_attributes(G, outcloseness_dict, 'out-closeness')
print("out-Closeness centrality:", outcloseness_dict)

# in-Katz centrality
inkatz_dict = nx.katz_centrality(G, weight= 'weight') 
nx.set_node_attributes(G, inkatz_dict, 'in-Katz')	
print("in-Katz centrality:", inkatz_dict)

# out-Katz centrality
outkatz_dict = nx.katz_centrality(G.reverse(), weight= 'weight') 
nx.set_node_attributes(G, outkatz_dict, 'out-Katz')	
print("out-Katz centrality:", outkatz_dict)

#in- Eigenvector centrality
ineigenvector_dict = nx.eigenvector_centrality(G, weight= 'weight') 
nx.set_node_attributes(G, ineigenvector_dict, 'In-eigenvector')	
print("in-Eigenvector centrality:", ineigenvector_dict)

#out- Eigenvector centrality
outeigenvector_dict = nx.eigenvector_centrality(G.reverse(), weight= 'weight') 
nx.set_node_attributes(G, outeigenvector_dict, 'Out-eigenvector')	
print("out-Eigenvector centrality:", outeigenvector_dict)


# in-Pagerank centrality
inpagerank_dict = nx.pagerank(G,max_iter=250, weight= 'weight') 
nx.set_node_attributes(G, inpagerank_dict, 'in-pagerank')	
print("in-pagerank:", inpagerank_dict)

# in-Pagerank centrality
outpagerank_dict = nx.pagerank(G.reverse(),max_iter=250, weight= 'weight') 
nx.set_node_attributes(G, outpagerank_dict, 'out-pagerank')	
print("out-pagerank:", outpagerank_dict)

#Hubs and Authorites
hubs_dict,auth_dict = nx.hits(G,max_iter=250) 
nx.set_node_attributes(G, hubs_dict, 'hubs')	
nx.set_node_attributes(G, auth_dict, 'auth')	
print("hubs:", hubs_dict)


#Clustering Coefficience
clustering_dict = nx.clustering(H, weight= 'weight') #I work with the undirected type 
nx.set_node_attributes(H, clustering_dict, 'Clust')	
print("clustering :", clustering_dict)


#Insert centralities to a worksheet
 #First row: Name of centrality
for i in range(15) :
    worksheet2.set_column(0, i, 20)   #Make space first row for names
 
worksheet2.write(0, 0, "Node", cell_format1)
worksheet2.write(0, 1, "In-Degree", cell_format1)
worksheet2.write(0, 2, "Out-Degree", cell_format1)
worksheet2.write(0, 3, "Betweenness", cell_format1)
worksheet2.write(0, 4, "In-Closness", cell_format1)
worksheet2.write(0, 5, "Out-Closness", cell_format1)
worksheet2.write(0, 6, "In-Katz", cell_format1)
worksheet2.write(0, 7, "Out-Katz", cell_format1)
worksheet2.write(0, 8, "In-Eigenvector", cell_format1)
worksheet2.write(0, 9, "Out-Eigenvector", cell_format1)
worksheet2.write(0, 10, "In-Pagerank", cell_format1)
worksheet2.write(0, 11, "Out-Pagerank", cell_format1)
worksheet2.write(0, 12, "Hubs", cell_format1)
worksheet2.write(0, 13, "Authorites", cell_format1)
worksheet2.write(0, 14, "Clustering Coef", cell_format1)

#Insert Centralities of each node per row
row = 0
col = 0

for key in indegree_dict.keys():
    row += 1
    worksheet2.write(row, col, labels_dict.get(key))
    worksheet2.write(row, col + 1, indegree_dict.get(key))
    worksheet2.write(row, col + 2, outdegree_dict.get(key))
    worksheet2.write(row, col + 3, betweenness_dict.get(key))
    worksheet2.write(row, col + 4, incloseness_dict.get(key))
    worksheet2.write(row, col + 5, outcloseness_dict.get(key))
    worksheet2.write(row, col + 6, inkatz_dict.get(key))
    worksheet2.write(row, col + 7, outkatz_dict.get(key))
    worksheet2.write(row, col + 8, ineigenvector_dict.get(key))
    worksheet2.write(row, col + 9, outeigenvector_dict.get(key))
    worksheet2.write(row, col + 10, inpagerank_dict.get(key))
    worksheet2.write(row, col + 11, outpagerank_dict.get(key))
    worksheet2.write(row, col + 12, hubs_dict.get(key))
    worksheet2.write(row, col + 13, auth_dict.get(key))
    worksheet2.write(row, col + 14, clustering_dict.get(key))
    

################ plot G based on centralities ##############
def draw(G, pos, measures, measure_name):
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=250, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=list(measures.keys()))
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))
    # labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos)
    
    plt.savefig(name + "_" + measure_name)   #save the plot image as a png file
    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.show()

draw(G, pos, indegree_dict, 'In-degree Centrality')
plt.show()

draw(G, pos, outdegree_dict, 'Out-degree Centrality')
plt.show()

draw(G, pos, incloseness_dict, 'in-Closeness Centrality')
plt.show()

draw(G, pos, outcloseness_dict, 'out-Closeness Centrality')
plt.show()

draw(G, pos, betweenness_dict, 'Betweenness Centrality')
plt.show()

draw(G, pos, inkatz_dict, 'in-Katz Centrality')
plt.show()

draw(G, pos, outkatz_dict, 'out-Katz Centrality')
plt.show()

draw(G, pos, ineigenvector_dict, 'in-eigenvector Centrality')
plt.show()

draw(G, pos, outeigenvector_dict, 'out-eigenvector Centrality')
plt.show()

draw(G, pos, inpagerank_dict, 'in-Pagerank Centrality')
plt.show()

draw(G, pos, outpagerank_dict, 'out-Pagerank Centrality')
plt.show()

draw(G, pos, hubs_dict, 'Hubs Centrality')
plt.show()

draw(G, pos, auth_dict, 'Authorities Centrality')
plt.show()

draw(G, pos, clustering_dict, 'Clustering Centrality')
plt.show()
#################  Second table #######################
N=G.order()
#in-degree centralization
indegrees = indegree_dict.values()
max_in = max(indegrees)
indeg_centralization = (N*max_in - sum(indegrees))/(N-1)**2

#out-degree centralization
outdegrees = outdegree_dict.values()
max_in1 = max(outdegrees)
outdeg_centralization = (N*max_in1 - sum(outdegrees))/(N-1)**2

#betweeness centralization
beetdegrees = betweenness_dict.values()
max_in2 = max(beetdegrees)
beet_centralization = (N*max_in2 - sum(beetdegrees))/((N-1)*(N-2)/(N-1))

#closeness centralization
closdegrees = incloseness_dict.values()
max_in3 = max(closdegrees)
clos_centralization = (N*max_in3 - sum(closdegrees))/((1-(1/N))*(N-1))

#cluster centralization
clustdegrees = clustering_dict.values()
max_in4 = max(clustdegrees)
clust_centralization = (N*max_in4 - sum(clustdegrees))/(N-1)**2

#In-degree entropy
indegseries=pd.pandas.Series(indegree_dict)  #transform it to a pd.series for entropy calculation
inp_data= indegseries.value_counts()/len(indegseries) # calculates the probabilities
entropy_in=sc.stats.entropy(inp_data)

#out-degree entropy
outdegseries=pd.pandas.Series(outdegree_dict)  #transform it to a pd.series for entropy calculation
p_data= outdegseries.value_counts()/len(outdegseries) # calculates the probabilities
entropy_out=sc.stats.entropy(p_data)  # input probabilities to get the entropy 

#request for an Erdos Renyi random graph with same number of nodes with G 
#and same number of edges

G_Random=nx.gnm_random_graph(len(G.nodes),len(G.edges))

# calculate Average shortest path length 
Path_Lenght_GRand=nx.average_shortest_path_length(G_Random)

#Calculate Clustering Coefficience
Clust_GRand=nx.average_clustering(G_Random)

#Calculate small-worldness of the network
small_worldness=(Clust_G/Clust_GRand)/(Path_Lenght_G/Path_Lenght_GRand)


worksheet3.set_column(0, 0, 30)

worksheet3.write(0,0,"In-Degree Centrality",cell_format1)
worksheet3.write(0,1,indeg_centralization, cell_format2)
worksheet3.write(1,0,"Out-Degree Centralization", cell_format1)
worksheet3.write(1,1,outdeg_centralization, cell_format2)
worksheet3.write(2,0,"Betweeness Centralization", cell_format1)
worksheet3.write(2,1,beet_centralization, cell_format2)
worksheet3.write(3,0,"Closeness Centralization", cell_format1)
worksheet3.write(3,1,clos_centralization, cell_format2)
worksheet3.write(4,0,"Clustering Centralization", cell_format1)
worksheet3.write(4,1,clust_centralization, cell_format2)
worksheet3.write(5,0,"small worldness", cell_format1)
worksheet3.write(5,1,small_worldness,cell_format2)
worksheet3.write(6,0,"in-Degree Entropy", cell_format1)
worksheet3.write(6,1,entropy_in, cell_format2)
worksheet3.write(7,0,"out-Degree Entropy", cell_format1)
worksheet3.write(7,1,entropy_out, cell_format2)

workbook.close()