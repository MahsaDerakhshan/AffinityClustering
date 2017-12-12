import sys
import argparse
import findspark
findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')
from pyspark import SparkContext
from basic import Edge, GroupedEdge
import math
import random
import simplejson
from itertools import groupby

class ArrayUnionFind:
    def __init__(self, S):
        self.group = dict((s,s) for s in S)
        self.size = dict((s,1) for s in S)
        self.items = dict((s,[s]) for s in S)

    def find(self, s):    
    #Return the id for the group containing s
        return self.group[s]

    def union(self, a,b):
    #Union the two sets a and b
        assert a in self.items and b in self.items
        if self.size[a] > self.size[b]:
            a,b = b,a
        for s in self.items[a]:
            self.group[s] = b
            self.items[b].append(s)
        self.size[b] += self.size[a]
        del self.size[a]
        del self.items[a]

    def get_items(self):
        return list(self.items.keys())


    def get_partitions(self):
        return list(self.items.values())


class Affinity:
    def __init__(self, arr, k):
        self.k = k
        self.E = [(a[0], a[1], a[2]) for a in arr]
        vertices = []
        for e in self.E:
            vertices.append(e[0])
            vertices.append(e[1])
        self.V = set(vertices)
        self.UF = ArrayUnionFind(self.V)
        self.closet_neighbors={}
        self.merged=set()

    def merge_with_closet_neighbor(self, v):
        if v in self.merged:
            return 
        if self.closet_neighbors[self.closet_neighbors[v]] is v:
            self.UF.union(self.UF.find(v), self.UF.find(self.closet_neighbors[v]))
            self.merged.add(v)
            self.merged.add(self.closet_neighbors[v])
        else:
            self.merge_with_closet_neighbor(self.closet_neighbors[v])
            self.UF.union(self.UF.find(v), self.UF.find(self.closet_neighbors[v]))
            self.merged.add(v)
        return

    def clustering(self):
        number_of_clusters = len(self.V)
        vertices = self.V
        while number_of_clusters > self.k:
            self.closet_neighbors = {}
            self.merged = set()
            d = []
            # adjacency list
            for e in self.E:
                d.append((e[0], e))
                d.append((e[1], (e[1], e[0], e[2])))
            min_edges=[]
            d = sorted(d, key=lambda x: x[0])
            for vetex_key, group in  groupby(d , lambda x: x[0]):
                edge_group = [e[1] for e in group]
                min_edges.append(min(edge_group, key = lambda e: e[2]))
            for edge in min_edges:
                self.closet_neighbors[edge[0]] = edge[1]
            for v in vertices:
                if v not in self.merged:
                    self.merge_with_closet_neighbor(v)

            #update edges
            new_edges = []
            for e in self.E:
                if self.UF.find(e[0]) !=  self.UF.find(e[1]):
                    new_edges.append((self.UF.find(e[0]), self.UF.find(e[1]), e[2]))
            self.E = new_edges

            #update vertices
            vertices = set(self.UF.get_items())
            number_of_clusters = len(self.UF.get_items())
        return self.UF.get_partitions()


def MST(arr):
    mst=[]
    E = [(a[0], a[1], a[2]) for a in arr]
    E = sorted(E, key= lambda x: x[2])
    V=[]
    for e in E:
        V.append(e[0])
        V.append(e[1])
    UF = ArrayUnionFind(set(V))
    for e in E:
        u_group = UF.find(e[0])
        v_group = UF.find(e[1])
        # if u,v are in different components
        if u_group != v_group:
            mst.append(e)
            UF.union(u_group, v_group)
    return mst

def partitioning1(x, k):
    edges=x[1]
    out = []
    partitionKey=random.randrange(0, k)
    for e in edges:
        out.append((e[1],(partitionKey, e)))
    return out

def partitioning2(x, k):
    edges=x[1]
    out = []
    partitionKey = random.randrange(0, k)
    for e in edges:
        firstPartiton=e[0]
        edge=e[1]
        out.append(((firstPartiton, partitionKey), edge))
    return out

def edge(string):
    splitted = string.split(',')
    return [int(splitted[0]), int(splitted[1]), float(splitted[2])]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Affinity custering inputs')
    parser.add_argument('--inp_path', type = str , required=True,
                        help="Input path")
    # To read multiple files use regex. For instance, inp_path=input/part-0** reads files part-000, ..., part-099
    parser.add_argument('--eps', type = float , default= 0.05,
                        help="Space per machine is at least n^(1+eps) where n is the number of data points")
    parser.add_argument('--out_path', type = str , required=True,
                        help="Output path")
    parser.add_argument('--k', type = int , required=True,
                        help="number of clusters")

    sc = SparkContext(appName="Affinity")
    args = parser.parse_args()
    inp = sc.textFile(args.inp_path)
    edges = inp.map(lambda x: edge(x))
    vertices = edges.flatMap(lambda x: [x[0], x[1]]).distinct()
    n = vertices.count()
    m = edges.count()
    eps= args.eps
    c = math.ceil(math.log(m))/math.ceil(math.log(n))-1
    while(c>eps):
        k = math.floor(n**((c-eps)/2))
        c = math.ceil(math.log(m))/math.ceil(math.log(n))-1
        keyedEdges = edges.map(lambda x: (x[0], x))
        half_partitioning = keyedEdges.groupByKey().flatMap(lambda x: partitioning1(x,k))
        full_partionining = half_partitioning.groupByKey().flatMap(lambda x: partitioning2(x,k))
        edges = full_partionining.groupByKey().flatMap(lambda x: MST(x[1]))
        m = edges.count()
    AF = Affinity(edges.collect(), args.k)
    clusters = AF.clustering()
    f = open(args.out_path, 'w')
    simplejson.dump(clusters, f)
    print("Done")
