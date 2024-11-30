# Lab 2: Using OpenStreetMap for Autonomous Vehicle Global and Local Planners
## Getting used to the Osmnx library
The first part of the lab involved getting to know the Osmnx library to use the data from Open Street Maps to compute paths between geographical points, using the built-in functions from osmnx. Then plotting the same path manually importing the nodes and edges from the Map and adjusting the lengths of the edges to provide a faithful geographical representation of the path. This was done taking into account the planar projection of the considered region (in our case from the city of Nantes), using the _Lambert 93_ conic projection:  
![image](https://github.com/user-attachments/assets/6be23a95-8003-491e-a390-771e0b1df53f)
![image](https://github.com/user-attachments/assets/7e0efc6f-43ec-4089-b973-ade55c869aef)

## Implementing a global planner
This second task was decoupled from the first one, here it involved a simple graph that was used as a benchmark for the implementation of an A* global planning algorithm, described in the `custom_a_star` function, coupled with the `custom_heuristic` function, in our case it was simply the air distance between starting and ending point.
![image](https://github.com/user-attachments/assets/ea8a7127-c0d2-4527-838a-e034287ecaf7)
![image](https://github.com/user-attachments/assets/0d75b4de-09ba-40f8-a2b5-b837803cfa0b)
## Implementing a local planner
Using as a foundation, the global planner developed in the second task, the objective of this last exercise was to create a _local path planner_ that could avoid potential obstacles, to do so it was chosen an algorithm based on the **dynamic window approach**






