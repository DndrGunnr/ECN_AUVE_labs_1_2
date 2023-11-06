import numpy as np
import open3d as o3d

import os
from os import path as osp

import pickle
import glob
import matplotlib.pyplot as plt

from labs.lab1.geometry import make_tf, apply_tf

# ---------------------------------------------
# --------------DO NOT MODIFY -----------------
CLASS_NAMES = ['car','truck','motorcycle', 'pedestrian']
CLASS_COLORS = plt.cm.rainbow(np.linspace(0, 1, len(CLASS_NAMES)))[:, :3]
CLASS_NAME_TO_COLOR = dict(zip(CLASS_NAMES, CLASS_COLORS))
CLASS_NAME_TO_INDEX = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))

# Path extraction
root_path = "/home/user/ECN_AUVE_labs/scenario1"

scenario = "Town01_type001_subtype0001_scenario00003"  


file_list = glob.glob(osp.join(root_path,
                                        'ego_vehicle', 'label', scenario) + '/*')
frame_list = []

with open(osp.join(root_path, "meta" ,scenario+ '.txt'), 'r') as f:
            lines = f.readlines()
line = lines[2]
agents =  [int(agent) for agent in line.split()[2:]]

for file_path in file_list:
    frame_list.append(file_path.split('/')[-1].split('.')[0])
frame_list.sort()
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
    lidar_data = np.load(root_path +  '/' + actor + '/lidar01/' + scenario +'/' + frame + '.npz')['data']
    lidar_T_actor = get_sensor_T_actor(actor, n_frame)
    lidar_data_actor = apply_tf(lidar_T_actor, lidar_data) #in actor frame
 
    return lidar_data_actor

def get_available_point_clouds(n_frame, actors):
    ego_to_world = get_actor_T_world(actors[0], n_frame)
    merged_pc = get_point_cloud(n_frame, actors[0]) #in ego frame
    # merged_pc = apply_tf(ego_to_world, merged_pc) #in world frame

    for actor in actors[1:]:
        actor_to_world = get_actor_T_world(actor, n_frame)
        

        lidar_data_actor = get_point_cloud(n_frame, actor) #in actor frame
        lidar_data_world = apply_tf(np.linalg.inv(ego_to_world) @ actor_to_world , lidar_data_actor) #in ego frame
        merged_pc = np.concatenate((merged_pc,lidar_data_world), axis=0)
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
         
    boxes = get_boxes_in_sensor_frame(n_frame, actor)
    boxes = np.array(boxes).reshape(-1,8) # get boxes in global frame (N,8)

    # TODO: map `boxes` from sensor frame to actor frame

    sensor_T_car = get_sensor_T_actor(actor, n_frame)  #TODO  np.eye(4)
    boxes_in_actor_frame = apply_tf(sensor_T_car, boxes)

    boxes = np.concatenate((boxes_in_actor_frame, boxes[:,3:]), axis=1)


    return boxes

def get_available_boxes_in_actor_frame(n_frame, actors):
    boxes = get_boxes_in_actor_frame(n_frame, actors[0]) #in ego frame
    boxes = np.array(boxes).reshape(-1,8)
    ego_to_world = get_actor_T_world(actors[0], n_frame)
    available_boxes_in_world_frame = boxes

    for actor in actors[1:]:
        actor_boxes = get_boxes_in_actor_frame(n_frame, actor) #in ego frame
        boxes = np.array(actor_boxes).reshape(-1,8)
        # sensor_T_car = get_sensor_T_actor(actors[0], n_frame)  #TODO  np.eye(4)
        car_T_world = get_actor_T_world(actor, n_frame)
        boxes_in_actor_frame = apply_tf(np.linalg.inv(ego_to_world) @ car_T_world, boxes)
        yaws = actor_boxes[:,6] + np.arctan2(car_T_world[1,0], car_T_world[0,0]) - np.arctan2(ego_to_world[1,0], ego_to_world[0,0])
        yaws = yaws.reshape(-1,1)
        boxes = np.concatenate((boxes_in_actor_frame,actor_boxes[:,3:-2],yaws,actor_boxes[:,-1].reshape(-1,1)), axis=1)
        available_boxes_in_world_frame = np.concatenate((available_boxes_in_world_frame, boxes), axis=0)

    return boxes




