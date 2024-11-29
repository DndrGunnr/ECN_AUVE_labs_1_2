import numpy as np
import open3d as o3d
import math
import os
from os import path as osp

import pickle
import glob
import matplotlib.pyplot as plt

from geometry import make_tf, apply_tf

# ---------------------------------------------
# ---------------------------------------------
# --------DO NOT MODIFY BELOW THIS-------------
# ---------------------------------------------
# ---------------------------------------------

CLASS_NAMES = ['car','truck','motorcycle', 'pedestrian']
CLASS_COLORS = plt.cm.rainbow(np.linspace(0, 1, len(CLASS_NAMES)))[:, :3]
CLASS_NAME_TO_COLOR = dict(zip(CLASS_NAMES, CLASS_COLORS))
CLASS_NAME_TO_INDEX = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))

# Path extraction
root_path = r"C:\Users\enric\Desktop\open3d_lab\scenario1"

scenario = "Town01_type001_subtype0001_scenario00003"  


file_list = glob.glob(osp.join(root_path,
                                        'ego_vehicle', 'label', scenario) + '/*')
frame_list = []

with open(osp.join(root_path, "meta" ,scenario+ '.txt'), 'r') as f:
            lines = f.readlines()
line = lines[2]
agents =  [int(agent) for agent in line.split()[2:]]

for file_path in file_list:
    frame_list.append(file_path.split('/')[-1].split('.')[0].split('\\')[-1])
frame_list.sort()
# ---------------------------------------------
# ---------------------------------------------
# --------DO NOT MODIFY ABOVE THIS-------------
# ---------------------------------------------
# ---------------------------------------------

def get_actor_T_world(actor, n_frame):

    frame = frame_list[n_frame]
    with open(osp.join(root_path, actor ,'calib',scenario, frame + '.pkl'), 'rb') as f:
        calib_dict = pickle.load(f)
    actor_tf_world = np.array(calib_dict['ego_to_world'])
    lidar_tf_actor = np.array(calib_dict['lidar_to_ego'])
    
    tf =  lidar_tf_actor @ actor_tf_world 
    trans = tf[:3,3]
    if actor == 'infrastructure':
        trans[2] += 2.0
    rot = tf[:3,:3]

    return make_tf(trans,rot) 

def get_sensor_T_actor(actor, n_frame):
    frame = frame_list[n_frame]
    with open(osp.join(root_path, actor ,'calib',scenario, frame + '.pkl'), 'rb') as f:
        calib_dict = pickle.load(f)
    lidar_tf_actor = np.array(calib_dict['lidar_to_ego'])

    tf =  lidar_tf_actor 
    trans = tf[:3,3]
    rot = tf[:3,:3]

    return make_tf(trans,rot) 

def get_point_cloud(n_frame, actor):

    frame = frame_list[n_frame] 
    lidar_data = np.load(osp.join(root_path, actor, 'lidar01', scenario, frame + '.npz'))['data']
    lidar_T_actor = get_sensor_T_actor(actor, n_frame)
    lidar_data_actor = apply_tf(lidar_T_actor, lidar_data) #in actor frame
 
    return lidar_data_actor

def get_available_point_clouds(n_frame, actors):
    '''
    :param n_frame: 
    :param actors:
    :return: (N, 8) - x, y, z, l, w, h, yaw, class

    This function is used to get all point clouds available in ego frame
    '''
    ego_to_world = get_actor_T_world(actors[0], n_frame) # the transformation from ego frame to world frame
    # Get the transformation from world frame to ego frame
    world_to_ego = np.linalg.inv(ego_to_world)
    merged_pc = get_point_cloud(n_frame, actors[0]) #in ego frame

    # TODO: retrieve point clouds in actor frame for all actors and merge them into one point cloud in ego frame
 
    for actor in actors[1:]: 
        
        # Get the transformation from i-th actor frame to world frame
        actor_to_world = get_actor_T_world(actor, n_frame)

        # Get the transformation from i-th actor frame to ego frame
        actor_to_ego = np.dot(world_to_ego, actor_to_world)

        lidar_data_actor = get_point_cloud(n_frame, actor)
        lidar_data_ego = apply_tf(actor_to_ego, lidar_data_actor)

        merged_pc = np.concatenate((merged_pc, lidar_data_ego), axis=0)

        
    return merged_pc


def get_boxes_in_sensor_frame(n_frame, actor):
    frame = frame_list[n_frame] 
    with open(osp.join(root_path,  actor ,'label',scenario, frame + '.txt'), 'r') as f:
        lines = f.readlines()
    
    boxes = []
    for line in lines[1:]:
        line = line.split()
        if line[-1] == 'False':
            continue
        box = np.array([float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7]), CLASS_NAME_TO_INDEX[line[0]]])
        # cx, cy, cz, l, w, h, yaw, class
        boxes.append(box)
    return boxes

def get_boxes_in_actor_frame(n_frame, actor): # TODO
    '''
    :param n_frame: 
    :param actor:
    :return: (N, 8) - x, y, z, l, w, h, yaw, class

    This function is used to get boxes detected by the actor in actor frame
    '''

    boxes = get_boxes_in_sensor_frame(n_frame, actor)
    boxes = np.array(boxes).reshape(-1,8) #in sensor frame

        # Map boxes from sensor frame to actor frame
    sensor_T_actor = get_sensor_T_actor(actor, n_frame)
    for i in range(len(boxes)):
        box_center = np.append(boxes[i][:3], 1)  # (x, y, z, 1)
        box_center_actor_frame = sensor_T_actor @ box_center
        boxes[i][:3] = box_center_actor_frame[:3]
        
    return boxes

def get_available_boxes_in_ego_frame(n_frame, actors):
    '''
    :param n_frame: 
    :param actors: a list of actors, the first one is ego vehicle
    :return: (N, 8) - x, y, z, l, w, h, yaw, class

    This function is used to get all available boxes by the actors in ego frame
    '''

    ego_to_world = get_actor_T_world(actors[0], n_frame)
    world_to_ego = np.linalg.inv(ego_to_world)

    # Get boxes in ego frame
    boxes = get_boxes_in_actor_frame(n_frame, actors[0])
    boxes = np.array(boxes).reshape(-1, 8)
    available_boxes_in_ego_frame = boxes

    # Retrieve boxes in actor frame for all actors and transform them to ego frame
    for actor in actors[1:]:
        actor_to_world = get_actor_T_world(actor, n_frame)
        actor_to_ego = np.dot(world_to_ego, actor_to_world)

        boxes = get_boxes_in_actor_frame(n_frame, actor)
        boxes = np.array(boxes).reshape(-1, 8)

        for i in range(len(boxes)):
            box_center = np.append(boxes[i][:3], 1)  # (x, y, z, 1)
            box_center_ego_frame = actor_to_ego @ box_center
            boxes[i][:3] = box_center_ego_frame[:3]

            # Rotate from actor to ego frame
            yaw = boxes[i][6]
            rotation_matrix = actor_to_ego[:3, :3]
            yaw_ego_frame = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) + yaw
            boxes[i][6] = yaw_ego_frame

        available_boxes_in_ego_frame = np.concatenate((available_boxes_in_ego_frame, boxes), axis=0)

    return available_boxes_in_ego_frame



def filter_points(points: np.ndarray, range: np.ndarray):
    '''
    points: (N, 3) - x, y, z
    range: (6,) - xmin, ymin, zmin, xmax, ymax, zmax

    return: (M, 3) - x, y, z
    This function is used to filter points within the range
    '''
    # : filter points within the range
    xmin, ymin, zmin, xmax, ymax, zmax = range
    filtered_points = []
    for point in points:
        x, y, z = point
        if x > xmin and x < xmax and y > ymin and y < ymax and z > zmin and z < zmax:
            filtered_points.append(point)
    filtered_points = np.array(filtered_points)
    return filtered_points


def detection_obj(n_frame,actors,method):
    '''
    :param n_frame: 
    :param actors: a list of actors, the first one is ego vehicle
    :param method: a string, describes type of objects to segment out

    This function is used to get only the boxes specified by method
    '''
    irsu_boxes = get_available_boxes_in_ego_frame(n_frame, actors)
    car = 0.0
    truck = 1.0
    motorcycle = 2.0
    pedestrian = 3.0
    boxes_car = []
    boxes_truck = []
    boxes_motorcycle = []
    boxes_pedestrian = []

    # Filter the objects
    for box in irsu_boxes:
        if(car == box[7]):
            boxes_car.append(box)
        if(truck == box[7]):
            boxes_truck.append(box)
        if(motorcycle == box[7]):
            boxes_motorcycle.append(box)
        if(pedestrian == box[7]):
            boxes_pedestrian.append(box)

    # Choose the boxes list based on method
    if method == 'Car':
        boxes = np.array(boxes_car)
    elif method == 'Truck':
        boxes = np.array(boxes_truck)

    elif method == 'Motorcycle':
        boxes = np.array(boxes_motorcycle)
    elif method == 'Pedestrians':
        boxes = np.array(boxes_pedestrian)
    elif method == 'All':
        boxes = irsu_boxes
    return boxes
