# tfj add

import os
import torch
import numpy as np
import open3d as o3d
import torch.utils.data as data
from scipy.spatial.transform import Rotation as R
import open3d as o3d


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.0225):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def farthest_point_sampling(points, num_samples):
    num_points = points.shape[0]
    sampled_indices = np.zeros(num_samples, dtype=np.int32)
    distances = np.full(num_points, np.inf)

    start_index = 0
    sampled_indices[0] = start_index

    for i in range(1, num_samples):
        last_sampled_index = sampled_indices[i - 1]
        last_sampled_point = points[last_sampled_index]

        dist_to_last_sampled = np.linalg.norm(points - last_sampled_point, axis=1)

        distances = np.minimum(distances, dist_to_last_sampled)

        next_sampled_index = np.argmax(distances)
        sampled_indices[i] = next_sampled_index

    return points[sampled_indices]

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ToolsDataset(data.Dataset):
    def __init__(self, args):
        super(ToolsDataset, self).__init__()

        self.input_num = args.input_num
        self.args = args
        # input and gt: (b, n, 3) radius: (b, 1)
        self.datas = []
        self.names = []
        print('Loading and processing point cloud, waiting ...')
        tmp_datas = []
        tmp_names = []
        load_data_path = os.path.join(args.file_path,'pc_{}pts.npz'.format(self.input_num))
        if os.path.exists(load_data_path):
            tdatas = np.load(load_data_path,allow_pickle=True)['data']
            tnames = np.load(load_data_path,allow_pickle=True)['name']
            for iii in range(len(tdatas)):
                tmp_datas.append(tdatas[iii])
                tmp_names.append(str(tnames[iii]))
            
        for root, dirs, files in os.walk(args.file_path):
            for file in files:
                if file.endswith('pcd'):
                    tmp_name = os.path.join(root,file)
                    
                    if tmp_name in tmp_names:
                        self.datas.append(tmp_datas[tmp_names.index(tmp_name)])
                        self.names.append(tmp_name)
                        continue
                    
                    pcd=o3d.io.read_point_cloud(tmp_name)
                    input=np.asarray(pcd.points)
                    
                    lens = len(input)
        
                    if lens < self.input_num :
                        ratio = int(self.input_num /lens + 1)
                        tmp_input = np.tile(input, (ratio, 1))
                        input = tmp_input[:self.input_num ]
                    
                    if lens > self.input_num :
                        np.random.shuffle(input)
                        input = farthest_point_sampling(input,self.input_num)
                        
                    self.datas.append(input)
                    self.names.append(tmp_name) 
                    
        np.savez(load_data_path,data=self.datas,name=self.names)
        print('data lens: ',len(self.datas))
                  
        # calculate similarity
        load_data_sim_path = os.path.join(args.file_path,'pc_{}pts_sim.npz'.format(self.input_num))
        tmpex_name,tmpex_sim = [],[]
        if os.path.exists(load_data_sim_path): # load existing data
            tmpex_sim = np.load(load_data_sim_path,allow_pickle=True)['sim']
            tmpex_name = np.load(load_data_sim_path,allow_pickle=True)['name']
        is_calc_sim = False
        if len(tmpex_name) != len(self.names):
            is_calc_sim = True
        else:
            for ii in range(len(self.names)):
                if self.names[ii]!=str(tmpex_name[ii]):
                    is_calc_sim = True
                    break
        self.sim = tmpex_sim
        if is_calc_sim:
            l = len(self.datas)      
            self.sim = np.zeros((l,l))
            print('please wait, calculating point cloud similarity ... ')
            for i in range(l):
                for j in range(i+1,l):  
                    point_cloud1 = o3d.geometry.PointCloud()
                    point_cloud1.points = o3d.utility.Vector3dVector(pc_normalize(self.datas[i]))
                    point_cloud2 = o3d.geometry.PointCloud()
                    point_cloud2.points = o3d.utility.Vector3dVector(pc_normalize(self.datas[j])) 
                    mean_distance_t_s = np.mean(point_cloud1.compute_point_cloud_distance(point_cloud2)) #chamfer distance
                    self.sim[i,j] = mean_distance_t_s
                    self.sim[j,i] = mean_distance_t_s
            np.savez(load_data_sim_path,sim=self.sim,name=self.names)
        print('finished calculatpoint cloud similarity ')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        input = self.datas[index]
        if np.random.rand() >= 0.5:
            label = 1
        else:
            label = 0
        input1 = input
        if label == 1:
            msg = 'positive sample'
            input2 = jitter_point_cloud(input[None,...]).squeeze()
        else:
            if np.random.rand() >= 0.5:
                random_rotation = R.random()
                rotation_matrix = random_rotation.as_matrix()
                input2 = np.dot(rotation_matrix, input.T).T
                msg = 'rotated negative sample'
            else:
                sim = self.sim[index]
                min_idx = np.argsort(sim)
                neg_id = np.random.choice(min_idx[-100:])
                input2 = self.datas[neg_id]
                msg = 'other negative samples'
                
        input1 = pc_normalize(input1)
        input1 = torch.from_numpy(input1)
        input2 = pc_normalize(input2)
        input2 = torch.from_numpy(input2)
        
        return input1,input2,label,msg