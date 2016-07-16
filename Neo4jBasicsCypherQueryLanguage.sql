###############################################################
###															###
###															###
###   				getting started							###
###															###
###															###
###############################################################

-- toy example to get started

create (N1:ToyNode {name: 'Tom'}) - [:ToyRelation {relationship: 'knows'}] -> (N2:ToyNode {name: 'Harry'}),

(N2) - [:ToyRelation {relationship: 'co-worker'}] -> (N3:ToyNode {name: 'Julian', job: 'plumber'}),

(N2) - [:ToyRelation {relationship: 'wife'}] -> (N4:ToyNode {name: 'Michele', job: 'accountant'}),

(N1) - [:ToyRelation {relationship: 'wife'}] -> (N5:ToyNode {name: 'Josephine', job: 'manager'}),

(N4) - [:ToyRelation {relationship: 'friend'}] -> (N5)

;

-- View the resulting graph
match (n:ToyNode)-[r]-(m) return n, r, m

-- Delete all nodes and edges
match (n)-[r]-() delete n, r

-- Delete all nodes which have no edges
match (n) delete n

-- Delete only ToyNode nodes which have no edges
match (n:ToyNode) delete n

-- Delete all edges
match (n)-[r]-() delete r

-- Delete only ToyRelation edges
match (n)-[r:ToyRelation]-() delete r

--Selecting an existing single ToyNode node
match (n:ToyNode {name:'Julian'}) return n


###############################################################
###															###
###															###
###   				adding or modifying nodes				###
###															###
###															###
###############################################################

--Adding a Node Correctly
match (n:ToyNode {name:'Julian'})
merge (n)-[:ToyRelation {relationship: 'fiancee'}]->(m:ToyNode {name:'Joyce', job:'store clerk'})

--Correct your mistake by deleting the bad nodes and edge
match (n:ToyNode {name:'Joyce'})-[r]-(m) delete n, r, m

--Modify a Node’s Information
match (n:ToyNode) where n.name = 'Harry' set n.job = 'drummer'
match (n:ToyNode) where n.name = 'Harry' set n.job = n.job + ['lead guitarist']


###############################################################
###															###
###															###
###   				Importing data							###
###															###
###															###
###############################################################

--One way to "clean the slate" in Neo4j before importing (run both lines below):

match (n)-[r]-() delete n, r
match (a) delete a

--[NOTE: replace any spaces in your path with %20, "percent twenty" ]

--Example 1
LOAD CSV WITH HEADERS FROM "file:///c:/Users/Aroon/Documents/Kaggle/neo4jmoduledatasets/test.csv" AS line
MERGE (n:MyNode {Name:line.Source})
MERGE (m:MyNode {Name:line.Target})
MERGE (n) -[:TO {dist:line.distance}]-> (m)

--Example 2
LOAD CSV WITH HEADERS FROM "file:///c:/Users/Aroon/Documents/Kaggle/neo4jmoduledatasets/terrorist_data_subset.csv" AS row
MERGE (c:Country {Name:row.Country})
MERGE (a:Actor {Name: row.ActorName, Aliases: row.Aliases, Type: row.ActorType})
MERGE (o:Organization {Name: row.AffiliationTo})
MERGE (a)-[:AFFILIATED_TO {Start: row.AffiliationStartDate, End: row.AffiliationEndDate}]->(o)
MERGE(c)<-[:IS_FROM]-(a);


###############################################################
###															###
###															###
###   			Visualize the data - basic					###
###															###
###															###
###############################################################

-- view limited number of nodes
MATCH (n) RETURN n LIMIT 25

-- view all nodes and relations
match (n:MyNode)-[r]->(m)
return n, r, m

--Not all nodes will be visible in default setting
:config initialNodeDisplay: 658 -- change initial node display to a number of your choice

-- even after this not all nodes will be visible.
-- use inspect trick. modify <g transform = "translate(0,0)" to <g transform = "translate(0,0) scale (0.15)"

###############################################################
###															###
###															###
###   					Basic queries						###
###															###
###															###
###############################################################

--Counting the number of nodes
match (n:MyNode)
return count(n)

--Counting the number of edges
match (n:MyNode)-[r]->()
return count(r)

--Finding leaf nodes: find a node from n to m, where NOT (m goes from somewhere to somewhere)
match (n:MyNode)-[r:TO]->(m)
where not ((m)-->())
return m

--Finding root nodes:
match (m)-[r:TO]->(n:MyNode)
where not (()-->(m))
return m

--Finding triangles:
match (a)-[:TO]->(b)-[:TO]->(c)-[:TO]->(a)
return distinct a, b, c

--Finding 2nd neighbors of D:
--:TO*..2 returns 2 neighbors
match (a)-[:TO*..2]-(b) 
where a.Name='D'
return distinct a, b

--Finding the types of a node:
match (n)
where n.Name = 'Afghanistan'
return labels(n)

--Finding the label of an edge:
match (n {Name: 'Afghanistan'})<-[r]-()
return distinct type(r)

--Finding all properties of a node:
match (n:Actor)
return * limit 20

--Finding loops:
match (n)-[r]->(n)
return n, r limit 10

--Finding loops: (return explicit node traversal in rows) 
match p = (n)-[r*]->(n)
RETURN EXTRACT(n IN NODES(p)| n.Name) AS Paths

--Finding multigraphs: (multigraphs have more than one relationship between nodes
match (n)-[r1]->(m), (n)-[r2]-(m)
where r1 <> r2
return n, r1, r2, m limit 10

--Finding the induced subgraph given a set of nodes: (return subgraph containing a given set of nodes
match (n)-[r:TO]-(m)
where n.Name in ['A', 'B', 'C', 'D', 'E'] and m.Name in ['A', 'B', 'C', 'D', 'E']
return n, r, m

###############################################################
###															###
###															###
###   					Path Analytics						###
###															###
###															###
###############################################################

--Finding paths between specific nodes:
--if you use :TO you only get direct nodes. If you use :TO* ... you get all nodes.
match p=(a)-[:TO*]-(c) 
where a.Name='H' and c.Name='P'
return p limit 1


match p=(a)-[:TO*]-(c) 
where a.Name='H' and c.Name='P' 
return p 
order by length(p) asc limit 1

match p=(a)-[:TO*]->(c) 
where a.Name='H' and c.Name='P' 
return p 
order by length(p) asc limit 1

--Finding the length defined by hops between specific nodes:
match p=(a)-[:TO*]->(c)
where a.Name='H' and c.Name='P'
return length(p) limit 10

--Finding a shortest path between specific nodes:
match p=shortestPath((a)-[:TO*]->(c))
where a.Name='H' and c.Name='P'
return p, length(p) limit 10

--All Shortest Paths: (sometimes there may be more than one short path)
MATCH p = allShortestPaths((source)-[r:TO*]-(destination))
WHERE source.Name='A' AND destination.Name = 'P'
RETURN EXTRACT(n IN NODES(p)| n.Name) AS Paths

--All Shortest Paths with Path Conditions:
MATCH p = allShortestPaths((source)-[r:TO*]->(destination))
WHERE source.Name='A' AND destination.Name = 'P' AND LENGTH(NODES(p)) > 5
RETURN EXTRACT(n IN NODES(p)| n.Name) AS Paths,length(p)

--Diameter of the graph:
match (n:MyNode), (m:MyNode)
where n <> m
with n, m
match p=shortestPath((n)-[*]->(m))
return n.Name, m.Name, length(p)
order by length(p) desc limit 1

--Extracting and computing with node and properties:
match p=(a)-[:TO*]-(c)
where a.Name='H' and c.Name='P'
return extract(n in nodes(p)|n.Name) as Nodes, length(p) as pathLength,
reduce(s=0, e in relationships(p)| s + toInt(e.dist)) as pathDist //limit 1

--Dijkstra's algorithm for a specific target node:

MATCH path = shortestPath((from) - [r:TO*] - (to))
where from.Name = 'A' and to.Name = 'P'
WITH REDUCE(dist = 0, rel in rels(path) | dist + toInt(rel.dist)) AS distance, path
RETURN path, distance

MATCH path = shortestPath((from) - [r:TO*] - (to))
where from.Name = 'A' and to.Name = 'P'
WITH REDUCE(dist = 0, rel in rels(path) | dist + toInt(rel.dist)) AS distance, path
RETURN EXTRACT(n IN NODES(path)| n.Name) AS Paths, distance

--Dijkstra's algorithm SSSP: (single source shortest path
MATCH (from: MyNode {Name:'A'}), (to: MyNode),
path = shortestPath((from)-[:TO*]->(to))
WITH REDUCE(dist = 0, rel in rels(path) | dist + toInt(rel.dist)) AS distance, path, from, to
RETURN from, to, path, distance 
order by distance desc

MATCH (from: MyNode {Name:'A'}), (to: MyNode),
path = shortestPath((from)-[:TO*]->(to))
WITH REDUCE(dist = 0, rel in rels(path) | dist + toInt(rel.dist)) AS distance, path, from, to
RETURN from, to, EXTRACT(n IN NODES(path)| n.Name) AS Paths, distance 
order by distance desc

--Graph not containing a selected node:
match (n)-[r:TO]->(m)
where n.Name <> 'D' and m.Name <> 'D'
return n, r, m

--Shortest path over a Graph not containing a selected node:
match p=shortestPath((a {Name: 'A'})-[:TO*]-(b {Name: 'P'}))
where not('D' in (extract(n in nodes(p)|n.Name)))
return p, length(p)

--Graph not containing the immediate neighborhood of a specified node:
match (d {Name:'D'})-[:TO]-(b)
with collect(distinct b.Name) as neighbors
match (n)-[r:TO]->(m)
where
not (n.Name in (neighbors+'D'))
and
not (m.Name in (neighbors+'D'))
return n, r, m
;
match (d {Name:'D'})-[:TO]-(b)-[:TO]->(leaf)
where not((leaf)-->())
return (leaf)
;
match (d {Name:'D'})-[:TO]-(b)<-[:TO]-(root)
where not((root)<--())
return (root)

--Graph not containing a selected neighborhood:
match (a {Name: 'F'})-[:TO*..2]-(b)
with collect(distinct b.Name) as MyList
match (n)-[r:TO]->(m)
where not(n.Name in MyList) and not (m.Name in MyList)
return distinct n, r, m

--Find the outdegree of all nodes
match (n:MyNode)-[r]->()
return n.Name as Node, count(r) as Outdegree
order by Outdegree
union
match (a:MyNode)-[r]->(leaf)
where not((leaf)-->())
return leaf.Name as Node, 0 as Outdegree -- returns leaf node with 0 outdegree

--Find the indegree of all nodes
match (n:MyNode)<-[r]-()
return n.Name as Node, count(r) as Indegree
order by Indegree
union
match (a:MyNode)<-[r]-(root)
where not((root)<--())
return root.Name as Node, 0 as Indegree

--Find the degree of all nodes
match (n:MyNode)-[r]-()
return n.Name, count(distinct r) as degree
order by degree

--Find degree histogram of the graph
match (n:MyNode)-[r]-()
with n as nodes, count(distinct r) as degree
return degree, count(nodes) order by degree asc

--Save the degree of the node as a new node property
match (n:MyNode)-[r]-()
with n, count(distinct r) as degree
set n.deg = degree
return n.Name, n.deg

--Construct the Adjacency Matrix of the graph
match (n:MyNode), (m:MyNode)
return n.Name, m.Name,
case
when (n)-->(m) then 1
else 0
end as value

--Construct the Normalized Laplacian Matrix of the graph
match (n:MyNode), (m:MyNode)
return n.Name, m.Name,
case
when n.Name = m.Name then 1
when (n)-->(m) then -1/(sqrt(toInt(n.deg))*sqrt(toInt(m.deg)))
else 0
end as value