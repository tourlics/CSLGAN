# -*- coding: utf-8 -*-
"""MyProblem.py"""
import numpy as np
import geatpy as ea
import json
import os
import xml.etree.ElementTree as ET
import random
from tqdm import tqdm
import sys
# sys.stdout = None
sys.path.append("..")
from parse_label_json import  split_center_xy
from itertools import combinations


class DataLoader():
    def __init__(self, inputjson, idx_list) -> None:
        #以1开始
        self.safety_coff = [
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,243,27,81,81,81,27,81,27,3,27,9,3],
            [0,243,0,9,9,9,27,27,27,27,27,27,3,3],
            [0,27,9,0,81,81,27,9,243,27,81,9,9,9],
            [0,81,9,81,0,9,27,9,9,243,27,27,9,9],
            [0,81,9,81,9,0,27,9,27,243,27,27,9,3],
            [0,81,27,27,27,27,0,27,9,27,81,27,27,27],
            [0,27,27,9,9,9,27,0,9,9,27,3,9,9],
            [0,81,27,243,9,27,9,9,0,27,243,27,27,27],
            [0,27,27,27,243,243,27,9,27,0,27,81,27,27],
            [0,3,27,81,27,27,81,27,243,27,0,9,81,27],
            [0,27,27,9,27,27,27,3,27,81,9,0,9,9],
            [0,9,3,9,9,9,27,9,27,27,81,9,0,81],
            [0,3,3,9,9,3,27,9,27,27,27,9,81,0],
        ]
        self.cost_coff = [
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,243,9,81,81,81,3,243,9,3,3,9,3],
            [0,243,0,9,27,27,27,9,27,3,243,3,3,9],
            [0,9,9,0,243,243,81,3,9,27,243,3,3,3],
            [0,81,27,243,0,3,27,3,3,27,243,3,27,3],
            [0,81,27,243,3,0,9,3,3,27,243,3,27,3],
            [0,81,27,81,27,9,0,3,9,3,27,3,3,3],
            [0,3,9,3,3,3,3,0,3,3,27,3,3,3],
            [0,243,27,9,3,3,9,3,0,9,243,3,27,9],
            [0,9,3,27,27,27,3,3,9,0,3,3,9,27],
            [0,3,243,243,243,243,27,27,243,3,0,3,27,27],
            [0,3,3,3,3,3,3,3,3,3,3,0,9,3],
            [0,9,3,3,27,27,3,3,27,9,27,9,0,27],
            [0,3,9,3,3,3,3,3,9,27,27,3,27,0]
        ]
        self.noisy_coff = [0, 166, 0, 88, 319, 340, 162, 0, 273,0,0,80,0,0] 
        self.idx_list = idx_list 
        self.inputjson = inputjson
        self.rect_info, self.W, self.H = self.parse_input_json()
        self.var_dim = (len(idx_list) - sum(idx_list))*2 
        self.var_idx_global_map, dim = self.map_idx_vars() 
        self.global_idx_var_map = self.map_vars_idx()
        assert self.var_dim == dim

    def parse_input_json(self):
        rect_info = np.zeros((30, 6)).tolist() 
        with open(self.inputjson, encoding='utf-8') as f:
            data = json.load(f)
        W = data["W"]
        H = data["H"]
        for key in data.keys():
            if key[0] != "F": continue
            for rect in data[key]:
                global_idx = rect["global_idx"]
                rect_info[global_idx] = [key, rect["c"], rect["xwidth"], rect["yheight"], rect["left_top"], rect["right_bottom"]]
        return rect_info, W, H

    def map_idx_vars(self): 
        var_idx_global_map = {}
        new_idx = 0
        for idx, item in enumerate(self.idx_list):
            if item == 0:
                var_idx_global_map[new_idx] = idx # {x1: rect["global_idx"]}
                var_idx_global_map[new_idx+1] = idx
                new_idx += 2
        return var_idx_global_map, new_idx
    
    def map_vars_idx(self): 
        idx_map = {}
        new_idx = 0
        for idx, item in enumerate(self.idx_list):
            if item == 0:
                idx_map[idx] = new_idx 
                new_idx += 1
        return idx_map

    def get_lb_ub_of_vars(self):
        lb = []
        ub = []
        for idx in range(0,self.var_dim,2):
            label, center, xwidth, yheight, left_top, right_bottom = self.get_info_of_var_idx(idx)
            lb.append(xwidth/2+5) 
            lb.append(yheight/2+5) 
            ub.append(self.W-5-xwidth/2) 
            ub.append(self.H-5-yheight/2) 
        return lb, ub
    
    def get_info_of_var_idx(self, idx):
        label, center, xwidth, yheight, left_top, right_bottom = self.rect_info[self.var_idx_global_map[idx]]
        return label, center, xwidth, yheight, left_top, right_bottom


class MyProblem(ea.Problem):  
    def __init__(self, dataset, penalty_factor): 
        name = 'CSLGAN'  
        self.dataset = dataset
        self.penalty_factor = penalty_factor
        M = 4 
        maxormins = [1] * M  
        Dim = dataset.var_dim
        varTypes = [1] * Dim  

        lb, ub = dataset.get_lb_ub_of_vars() 

        assert len(lb) == len(ub) == Dim

        lbin = [0]*len(lb) 
        ubin = [0]*len(ub) 
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    
    def rectangle_overlap_penalty(self, rect1, rect2, penalty_factor):  
        center1, xwidth1, yheight1 = np.asarray(rect1[0]), np.asarray(rect1[1]), np.asarray(rect1[2])
        center2, xwidth2, yheight2 = np.asarray(rect2[0]), np.asarray(rect2[1]), np.asarray(rect2[2])
        
        if len(center2.shape) != 3: 
            center2 = center2.reshape((2, 1, 1)).repeat(center1.shape[1], axis=1) 
        np1 = np.abs(center1[0,:,:]-center2[0,:,:])-(xwidth1+xwidth2)/2
        np2 = np.abs(center1[1,:,:]-center2[1,:,:])-(yheight1+yheight2)/2
        npmax = np.maximum(np1, np2)
        penalty = np.zeros(npmax.shape)
        penalty[np.where(npmax<5)] = penalty_factor
        return penalty

    def aimFunc(self, pop):
        Vars = pop.Phen 
        xi = []
        yi = []
        for i in range(0,self.dataset.var_dim,2):
            xi.append(Vars[:, [i]]) 
            yi.append(Vars[:, [i+1]]) 
        assert len(xi) == len(yi)

        f1 = 0 
        f2 = 0 
        f4 = 0 
        penalty = np.zeros(xi[0].shape)
        penalty_factor = self.penalty_factor
        
        for i in range(len(xi)): 
            labeli,_,xwidthi,yheighti,_,_ = self.dataset.get_info_of_var_idx(i*2)
            for j,item in enumerate(self.dataset.idx_list):
                if item == 1: 
                    center = self.dataset.rect_info[j][1] 
                    labelj = self.dataset.rect_info[j][0] 
                    cur_dis = (xi[i] - center[0])**2 + (yi[i] - center[0])**2 + 1 
                    f4_Y = 5.548 * np.log( cur_dis ) - 1.042
                    f4 += np.power(10,0.01*(self.dataset.noisy_coff[int(labelj[1:])]-f4_Y))
                    # print(0.1*(self.dataset.noisy_coff[int(labelj[1:])]-f4_Y))
                    if j!=self.dataset.var_idx_global_map[i*2]:
                        f1 += ((xi[i] - center[0])**2 + (yi[i] - center[0])**2) * \
                            self.dataset.safety_coff[int(labeli[1:])][int(labelj[1:])]
                        f2 += ((xi[i] - center[0])**2 + (yi[i] - center[0])**2) * \
                            self.dataset.cost_coff[int(labeli[1:])][int(labelj[1:])]
                        xwidth = self.dataset.rect_info[j][2] 
                        yheight = self.dataset.rect_info[j][3] 
                        # print(i, self.dataset.var_idx_global_map[i*2],j)
                        penalty += self.rectangle_overlap_penalty([[xi[i], yi[i]], xwidthi, yheighti], \
                                                                [center, xwidth, yheight], penalty_factor)

        for i in range(len(xi)):
            labeli,_,xwidthi,yheighti,_,_ = self.dataset.get_info_of_var_idx(i*2)
            for j in range(len(xi)):
                if i == j: continue
                labelj,centerj,xwidthj,yheightj,_,_ = self.dataset.get_info_of_var_idx(j*2)
                f1 += ((xi[i] - xi[j])**2 + (yi[i] - yi[j])**2) * \
                        self.dataset.safety_coff[int(labeli[1:])][int(labelj[1:])]
                f2 += ((xi[i] - xi[j])**2 + (yi[i] - yi[j])**2) * \
                        self.dataset.cost_coff[int(labeli[1:])][int(labelj[1:])]
                # print(i,j)
                penalty += self.rectangle_overlap_penalty([[xi[i], yi[i]], xwidthi, yheighti], \
                                                          [[xi[j], yi[j]], xwidthj, yheightj], penalty_factor)
        
        f3 = 0
        rect_F2 = []
        rect_F1 = []
        rect_F8 = []
        for global_idx in range(len(self.dataset.idx_list)):
            label, center, _, _, _, _ = self.dataset.rect_info[global_idx]
            if self.dataset.idx_list[global_idx] == 1: 
                if label == "F2": rect_F2.append(np.asarray([center]).reshape(2,-1,1).repeat(xi[0].shape[0], axis=1))
                if label == "F1": rect_F1.append(np.asarray([center]).reshape(2,-1,1).repeat(xi[0].shape[0], axis=1))
                if label == "F8": rect_F8.append(np.asarray([center]).reshape(2,-1,1).repeat(xi[0].shape[0], axis=1))
            else : 
                var_idx = self.dataset.global_idx_var_map[global_idx]
                if label == "F2": rect_F2.append(np.asarray([xi[var_idx], yi[var_idx]]))
                if label == "F1": rect_F1.append(np.asarray([xi[var_idx], yi[var_idx]]))
                if label == "F8": rect_F8.append(np.asarray([xi[var_idx], yi[var_idx]]))
        vmove = 2.4/180*np.pi
        for rtF2 in rect_F2:
            for rtF1 in rect_F1:
                for rtF8 in rect_F8:
                    d28_pow2 = np.sum((rtF2-rtF8)**2, axis=0)
                    d12_pow2 = np.sum((rtF1-rtF2)**2, axis=0)
                    d18_pow2 = np.sum((rtF1-rtF8)**2, axis=0)
                    Twg = 2/vmove * np.arccos((d28_pow2 + d12_pow2 - d18_pow2)/(2 * np.sqrt(d12_pow2) * np.sqrt(d28_pow2)))
                    Tag = 2*np.abs(np.sqrt(d28_pow2)-np.sqrt(d12_pow2))/0.5
                    f3 += Twg + Tag
                    
        f = np.hstack([f1, f2, f3, f4]) + penalty
        # f = f1 + penalty
        pop.ObjV = f 
        

def generate_opt_xml(opt_var, dataset, originxml, optxml, objlabel):
    tree = ET.parse(originxml)
    root = tree.getroot()

    for layout in root.findall('layout'):
        for element in layout.findall('element'):
            global_idx = int(element.get('global_idx'))
            for idx in range(0,len(opt_var),2):
                if dataset.var_idx_global_map[idx] == global_idx:
                    center, xwidth, yheight = dataset.rect_info[global_idx][1], dataset.rect_info[global_idx][2], dataset.rect_info[global_idx][3]
                    polygon_x, polygon_y = split_center_xy([opt_var[idx], opt_var[idx+1]], xwidth, yheight)
                    element.set("polygon_x", polygon_x)  
                    element.set("polygon_y", polygon_y) 
    keyword = ET.SubElement(ET.SubElement(root, 'text'),'keyword')
    keyword.text = objlabel
    tree.write(optxml, encoding="utf-8", xml_declaration=True) 
            
def get_objlabels_from_minobjs(minobjs):
    obj_names = ["safety", "cost", "effiency", "noisy"]
    minobjs = np.asarray(minobjs)
    objlabels = [""]*minobjs.shape[0]
    for col_obj in range(minobjs.shape[1]):
        min_idx_per_col = np.argsort(minobjs[:,col_obj])[:int(minobjs.shape[0]/2)] 
        for item in min_idx_per_col:
            objlabels[item] += f"{obj_names[col_obj]} "
    for idx in range(len(objlabels)):
        objlabels[idx] = objlabels[idx][:-1] 
    return objlabels 

# def generate_lists(n, i, fixed_indices):
#     base_list = [0] * n
#     for index in fixed_indices:
#         base_list[index] = 1
#     remaining_ones = i
#     available_indices = [idx for idx in range(n) if idx not in fixed_indices]
#     indices_combinations = combinations(available_indices, remaining_ones)
    
#     result = []
#     for indices in indices_combinations:
#         lst = base_list[:]
#         for index in indices:
#             lst[index] = 1
#         result.append(lst)
    
#     return result

if __name__ == '__main__':

    dataroot = r""
    ann_aug_dir = os.path.join(dataroot, "ann_aug")
    originxml_dir = os.path.join(dataroot, "selected_xml")
    optxml_dir = os.path.join(dataroot, "opt_xml")
    os.makedirs(optxml_dir, exist_ok=True)

    scene_name_list = ["sample03"]

 
    for file in scene_name_list:
        scene_name = file.split(".")[0]
        print(f"*****processing {scene_name}...**********")
        originxml = os.path.join(originxml_dir, f"{scene_name}.xml")
        multiple_idx_list = [
            [1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,0,1,1,1],
            [1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1],
            [1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0]
        ]
        
        
        opt_xml_idx = 0

        for idx_list in tqdm(multiple_idx_list):
            # print(f"processing the {opt_xml_idx} of {scene_name}...")
            inputjson = os.path.join(ann_aug_dir, f"{scene_name}.json")
            originxml = os.path.join(originxml_dir, f"{scene_name}.xml")

            dataset = DataLoader(inputjson, idx_list)
            # print(dataset.rect_info)
            penalty_factor = 1e20

            problem = MyProblem(dataset, penalty_factor) 

            Encoding = 'RI' 
            NIND = 10000 
            Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) 
            population = ea.Population(Encoding, Field, NIND) 

            myAlgorithm = ea.moea_NSGA2_templet(problem, population) 

            myAlgorithm.MAXGEN = 1
            myAlgorithm.mutOper.F = 0.5 
            myAlgorithm.recOper.XOVR = 0.7

            res = ea.optimize(myAlgorithm, verbose=False, drawing=False, outputMsg=False, drawLog=False, saveFlag=False, dirName='result')
            
            if res["success"]:
                # print("optimizing......")
                minobjs = res["ObjV"]
                if np.any(minobjs >= penalty_factor):
                    print(f"WARNING!!!!!!: Rectangles may being overlapped!!!!!")
                    print("STOPPED OPTIMIZING!")
                    continue
                else:
                    opt_vars = res["Vars"]
                    print("minobjs", minobjs)
                    print("opt_vars",opt_vars)
                    num_of_solution = len(minobjs)
                    for i in range(num_of_solution):
                        optxml = os.path.join(optxml_dir, f"{scene_name}_{opt_xml_idx}.xml")
                        objlabels = get_objlabels_from_minobjs(minobjs)
                        generate_opt_xml(opt_vars[i], dataset, originxml, optxml, objlabels[i])
                        # print(f"{opt_xml_idx},......,{idx_list},{minobjs}")
                        opt_xml_idx += 1
            else:
                # print("ERROR: optimized failed")
                pass