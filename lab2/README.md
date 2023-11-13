# Lab 2: Using OpenStreetMap for Autonomous Vehicle Local Planners

## Objectives:
1. Understand the fundamentals of OpenStreetMap (OSM) maps.
2. Develop an understanding of local planners for autonomous vehicles.
3. Gain practical experience in designing, implementing, and testing local planners.

## Description:
This lab provides a comprehensive understanding and hands-on experience with OpenStreetMap maps and the development of local planners for autonomous vehicles. You will learn how to load and visualize OSM maps and implement pathfinding algorithms in a simulated environment. This experience is crucial for further study or work in the field of autonomous vehicle technology.

## Steps:
1. Load and Visualize OSM Maps:
   - Use the osmnx library to load an OSM map for a specified location.
   - Visualize the OSM graph, nodes, and edges for this location.
   - Select an area of interest on https://www.openstreetmap.org to work with.

2. Implement Global Path Finding:
   - Determine a path from an arbitrary origin to a destination using the nx library.
   - Plot this path on the OSM graph.
   - Analyze the chosen path for its feasibility and efficiency.

3. Implement a simplified Dynamic Window Approach (DWA) for local planning 
   - Implement cost functions 
   - Implement evolution model (based on the constant velocity bicycle model)

## Expected Learning Outcomes:
- Comprehensive understanding of OSM maps and their application in autonomous vehicle navigation.
- Ability to design, implement, and test local planners for autonomous vehicles in a simulated environment.
- Skills in analyzing and interpreting simulated navigation data.

## Ressources:
- networkX
- osmnx
- openstreetmap


## Grading Criteria
This lab is designed to bridge the gap between theoretical knowledge and practical application, preparing you for advanced studies or careers in autonomous vehicle technology.
Your lab submission will be graded based on the following criteria:
- Completion of the code and fulfillment of the specified tasks.
- Correct implementation of heuristic functions and obstacle avoidance logic.
- Successful testing of the code on the scenario.
- Code readability, comments, and documentation in the report.
- Adherence to good programming practices.

## Deadline
The deadline for this lab submission is the *4th of December (11:59 pm)* on Hippocampus website 
Upload the following completed files:

- lab2.ipynb
- osmnx_utils.py 
