import os
import numpy as np
import json
import torch
import csv
import math
import pandas as pd
import copy

class SkeletonLoaderTRI(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to data folder
        num_track: number of skeleton output
        pad_value: the values for padding missed joint
        repeat: times of repeating the dataset
    """
    def __init__(self, data_dir, num_track=1, repeat=1, num_keypoints=-1, 
                outcome_label='UPDRS_gait', missing_joint_val=0, csv_loader=False, 
                cache=False, layout='coco', flip_skels=False):
        self.data_dir = data_dir
        self.num_track = num_track
        self.num_keypoints = num_keypoints
        self.files = data_dir * repeat
        self.outcome_label = outcome_label
        self.missing_joint_val = missing_joint_val
        self.csv_loader = csv_loader
        self.interpolate_with_mean = False
        self.layout = layout
        self.class_dist = {}
        self.flip_skels = flip_skels

        if self.missing_joint_val == 'mean':
            self.interpolate_with_mean = True
            self.missing_joint_val = 0


        self.cache = cache
        self.cached_data = {}
        if self.cache:
            print("loading data to cache...")
            for index in range(self.__len__()):
                self.__getitem__(index)


    def __len__(self):
        if self.flip_skels:
            return len(self.files)*2
        return len(self.files)

    def __getitem__(self, index):
# {
    # "info":
        # {
            # "video_name": "skateboarding.mp4",
            # "resolution": [340, 256],
            # "num_frame": 300,
            # "num_keypoints": 17,
            # "keypoint_channels": ["x", "y", "score"],
            # "version": "1.0"
        # },
    # "annotations":
        # [
            # {
                # "frame_index": 0,
                # "id": 0,
                # "person_id": null,
                # "keypoints": [[x, y, score], [x, y, score], ...]
            # },
            # ...
        # ],
    # "category_id": 0,
# }
        if index in self.cached_data:
            # print("loading from cache: ", self.files[index])
            return self.cached_data[index]
        # print("loading the data from file")

        if index >= len(self.files):
            flip_index = index
            index = index - len(self.files)
            return_flip = True
        else:
            flip_index = index + len(self.files)
            return_flip = False


        if self.csv_loader:
            # print("getting itemmmmm", self.outcome_label)
            # print(self.files[index])
            file_index = index
            if index >= len(self.files):
                file_index = index - len(self.files)

            data_struct_interpolated = pd.read_csv(self.files[file_index])
            data_struct_interpolated.fillna(data_struct_interpolated.mean(), inplace=True)

            # print(data_struct_interpolated.head())

            data_struct = {} 
            with open(self.files[file_index]) as f:        
                data = csv.reader(f)
                csvreader = csv.DictReader(f)
                for row in csvreader:
                    for colname in row:
                        if colname not in data_struct:
                            try:
                                data_struct[colname] = [float(row[colname])]
                            except ValueError as e:
                                data_struct[colname] = [row[colname]]

                        else:
                            try:
                                data_struct[colname].append(float(row[colname]))
                            except ValueError as e:
                                data_struct[colname].append(row[colname])

            if self.layout == 'coco':
                num_kp = 17
                order_of_keypoints = ['Nose', 
                    'LEye', 'REye', 'LEar', 'REar',
                    'LShoulder', 'RShoulder',
                    'LElbow', 'RElbow', 
                    'LWrist', 'RWrist', 
                    'LHip', 'RHip',
                    'LKnee', 'RKnee',
                    'LAnkle', 'RAnkle',
                ]

            elif self.layout == 'coco_simplified_head':
                num_kp = 13
                order_of_keypoints = ['Nose', 
                    'LShoulder', 'RShoulder',
                    'LElbow', 'RElbow', 
                    'LWrist', 'RWrist', 
                    'LHip', 'RHip',
                    'LKnee', 'RKnee',
                    'LAnkle', 'RAnkle',
                ]
            else:
                raise ValueError(f"The layout {self.layout} does not exist")

            # print(data_struct)
            try:
                info_struct = {
                    "video_name": data_struct['walk_name'][0],
                    "resolution": [1920, 1080],
                    "num_frame": len(data_struct['time']),
                    "num_keypoints": num_kp,
                    "keypoint_channels": ["x", "y", "score"],
                    "version": "1.0"
                }
            except:
                print('data_struct', data_struct)            
                raise ValueError("something is wrong with the data struct", self.files[file_index])
            # order_of_keypoints = {'Nose', 
            #     'RShoulder', 'RElbow', 'RWrist', 
            #     'LShoulder', 'LElbow', 'LWrist', 
            #     'RHip', 'RKnee', 'RAnkle', 
            #     'LHip', 'LKnee', 'LAnkle', 
            #     'REye', 'LEye', 'REar', 'LEar'}



            annotations = []
            annotations_flipped = []

            for ts in range(len(data_struct['time'])):
                ts_keypoints, ts_keypoints_flipped = [], []
                for kp_num, kp in enumerate(order_of_keypoints):
                    if kp == "Neck":
                        RShoulder = [data_struct['RShoulder_x'][ts], data_struct['RShoulder_y'][ts], data_struct['RShoulder_conf'][ts]]   
                        LShoulder = [data_struct['LShoulder_x'][ts], data_struct['LShoulder_y'][ts], data_struct['LShoulder_conf'][ts]]   
                        # print(RShoulder, LShoulder)
                        x = ( RShoulder[0] +  LShoulder[0] ) / 2
                        y = ( RShoulder[1] +  LShoulder[1] ) / 2
                        try:
                            conf = ( RShoulder[2] +  LShoulder[2] ) / 2
                        except:
                            conf = 0
                    else:
                        x = data_struct[kp + '_x'][ts]          
                        y = data_struct[kp + '_y'][ts]          
                        conf = data_struct[kp + '_conf'][ts]      
                    

                        # missing actual joint coordinates
                        try:
                            x = float(x)
                            y = float(y)
                        except:
                            if self.interpolate_with_mean:
                                x = data_struct_interpolated[kp + '_x'][ts]          
                                y = data_struct_interpolated[kp + '_y'][ts]          
                            else:           
                                x = self.missing_joint_val
                                y = self.missing_joint_val
                            
                            if isinstance(x, str):
                                x = self.missing_joint_val

                            if isinstance(y, str):
                                y = self.missing_joint_val

                        if math.isnan(x) or math.isnan(y):
                            x = self.missing_joint_val
                            y = self.missing_joint_val
                            # if self.interpolate_with_mean:
                            #     x = data_struct_interpolated[kp + '_x'][ts]          
                            #     y = data_struct_interpolated[kp + '_y'][ts]          
                            # else:           
                            #     x = self.missing_joint_val
                            #     y = self.missing_joint_val


                        # print("kp, x, y, x is nan", kp, type(x), y)
                        # print('isnan',  math.isnan(x))

                        # Flip the left and right sides (flipping x)
                        if kp_num == 0: # Nose isn't flipped
                            x_flipped = x
                        else:
                            cur_side = kp[0]
                            if cur_side.upper() == "L":
                                kp_other_side = "R" + kp[1:]
                            elif cur_side.upper() == "R":
                                kp_other_side = "L" + kp[1:]
                            else:
                                raise ValueError("cant flip: ", kp)
                            x_flipped = data_struct[kp_other_side + '_x'][ts]  

                            # missing actual joint coordinates
                            try:
                                x_flipped = float(x_flipped)
                            except:
                                if self.interpolate_with_mean:
                                    x_flipped = data_struct_interpolated[kp_other_side + '_x'][ts]          
                                else:           
                                    x_flipped = self.missing_joint_val




                            if math.isnan(x_flipped):
                                x_flipped = self.missing_joint_val


                    # missing confidence = 0
                    try:
                        conf = float(conf)
                    except:                    
                        conf = 0


                    if math.isnan(conf):
                        conf = 0

 


                    ts_keypoints.append([x, y, conf])
                    ts_keypoints_flipped.append([x_flipped, y, conf])

                cur_ts_struct = {'frame_index': ts,
                                'id': 0, 
                                'person_id': 0,
                                'keypoints': ts_keypoints}

                cur_ts_struct_flipped = {'frame_index': ts,
                                'id': 0, 
                                'person_id': 0,
                                'keypoints': ts_keypoints_flipped}

                annotations.append(cur_ts_struct)
                annotations_flipped.append(cur_ts_struct_flipped)

            # print(annotations) 
            outcome_cat = data_struct[self.outcome_label][0]
            try:
                outcome_cat = float(outcome_cat)
                outcome_cat = int(outcome_cat)   
            except:
                outcome_cat = -1

            if outcome_cat in self.class_dist:
                self.class_dist[outcome_cat] += 1
            else:
                self.class_dist[outcome_cat] = 1

            # print("annotations: ",  annotations)
            # raise ValueError("ok")
            data = {'info': info_struct, 
                        # 'annotations': annotations,
                        'category_id': outcome_cat}
        
        else: # original loader 
            with open(self.files[index]) as f:
                data = json.load()
        # # print("we got: ", data_arr[0])

        info = data['info']
        # annotations = data['annotations']
        # annotations_flipped = data['annotations_flipped']
        num_frame = info['num_frame']
        num_keypoints = info[
            'num_keypoints'] if self.num_keypoints <= 0 else self.num_keypoints
        channel = info['keypoint_channels']
        num_channel = len(channel)

        # # get data
        data['data'] = np.zeros(
            (num_channel, num_keypoints, num_frame, self.num_track),
            dtype=np.float32)

        for a in annotations:
            person_id = a['id'] if a['person_id'] is None else a['person_id']
            frame_index = a['frame_index']
            if person_id < self.num_track and frame_index < num_frame:
                data['data'][:, :, frame_index, person_id] = np.array(
                    a['keypoints']).transpose()
        data['data_flipped'] = np.zeros(
                (num_channel, num_keypoints, num_frame, self.num_track),
                dtype=np.float32)

        for a in annotations_flipped:
            person_id = a['id'] if a['person_id'] is None else a['person_id']
            frame_index = a['frame_index']
            if person_id < self.num_track and frame_index < num_frame:
                data['data_flipped'][:, :, frame_index, person_id] = np.array(
                    a['keypoints']).transpose()

          
        flipped_data = copy.deepcopy(data)
        temp_flipped = flipped_data['data_flipped']
        flipped_data['data_flipped'] = flipped_data['data']
        flipped_data['data'] = temp_flipped
        flipped_data['name'] = self.files[file_index] + "_flipped"

        data['name'] = self.files[file_index]

        if self.cache:
            self.cached_data[index] = data

            if self.flip_skels:    
                self.cached_data[flip_index] = flipped_data


        if self.flip_skels and return_flip:
            return flipped_data
        # print(data['data'])
        # data.pop('annotations', None)
        # print(data)
        # raise ValueError("stop")
        return data

