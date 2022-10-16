from distutils.command.install_data import install_data
import io
import numpy as np 
from collections import Counter
import time 
import traceback
import os
from tqdm import tqdm
import sys 
import shutil
sys.setrecursionlimit(10000)



def iou_matching(data_dir):
    
    ## Get predictions from EffLPS
    
    path = os.path.join(data_dir, "08/predictions")
    label_paths =  [os.path.join(path, file)
                       for file in sorted(os.listdir(path))]

    ## Sort flowdataset
    path_root = os.path.join("_output/flow_dataset", "sequences/08/velodyne")
    try:
        os.makedirs(os.path.join(path_root, "orig"))
        os.makedirs(os.path.join(path_root, "prime"))
    except:
        pass
    velodyne_paths = [os.path.join(path_root, file) for file in sorted(os.listdir(path_root))]
    for path in velodyne_paths:
        if "orig" in path:
            print(os.path.join(path_root, "orig/%s" % os.path.basename(path)))
            try:
                shutil.copyfile(path, os.path.join(path_root, "orig/%s" % os.path.basename(path)))
            except:
                pass
        elif "prime" in path: 
            try:
                shutil.copyfile(path, os.path.join(path_root, "prime/%s" % os.path.basename(path)))
            except:
                pass

    path = os.path.join("_output/flow_dataset", "sequences/08/velodyne/prime")
    pc_prime_paths = [os.path.join(path, file)
                       for file in sorted(os.listdir(path))]
    
    path = os.path.join("_output/flow_dataset", "sequences/08/velodyne/orig")
    pc_orig_paths = [os.path.join(path, file)
                       for file in sorted(os.listdir(path))]

    if not os.path.isdir("./EfficientLPS/tmpDir/08/orig"):
        os.makedirs("./EfficientLPS/tmpDir/08/orig")

    if not os.path.isdir("./EfficientLPS/tmpDir/08/prime"):
        os.makedirs("./EfficientLPS/tmpDir/08/prime")

    for path in label_paths:
        if "orig" in path:
            dst = os.path.join("./EfficientLPS/tmpDir/08/orig", os.path.basename(path))
            shutil.copyfile(path, dst)

        if "prime" in path:
            dst = os.path.join("./EfficientLPS/tmpDir/08/prime", os.path.basename(path))
            shutil.copyfile(path, dst)

    path = os.path.join(data_dir, "08/orig")
    orig_paths =  [os.path.join(path, file)
                       for file in sorted(os.listdir(path))]
    
    path = os.path.join(data_dir, "08/prime")
    prime_paths =  [os.path.join(path, file)
                       for file in sorted(os.listdir(path))]
    
    match_list = []
    
    for iter in range(0, 13, 1):
        print("***ITERATION %d ***" % iter)
        match_label = []
        
        label_orig = np.fromfile(orig_paths[iter +1], dtype=np.int32).reshape((-1))
        label_prime = np.fromfile(orig_paths[iter], dtype=np.int32).reshape((-1))



        label_sem_orig = label_orig & 0xFFFF
        label_sem_prime = label_prime & 0xFFFF

        label_inst_orig = label_orig >> 16
        label_inst_prime = label_prime >> 16

        pc_orig = np.fromfile(pc_orig_paths[iter + 1], dtype=np.float32).reshape((-1, 3))[:, :3]
        pc_prime = np.fromfile(pc_prime_paths[iter], dtype=np.float32).reshape((-1, 3))[:, :3]

        


        ## get instances of certain semantic classes: 
        instance_classes = [0, 1, 40, 44, 48, 49, 50, 51, 52, 60, 70, 71, 72]
        semantic_classes = [10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 80, 81, 99, 252, 253, 254, 255, 256, 257, 258, 259]
        

        semantic_idx_orig = np.unique(label_orig[np.where(np.isin((label_orig & 0xFFFF), instance_classes))] >> 16)
        semantic_idx_prime = np.unique(label_prime[np.where(np.isin((label_prime & 0xFFFF), instance_classes))] >> 16)

        print("Removed Unlabeled Noise for Orig is: ", semantic_idx_orig)
        print("Removed Unlabeled Noise for Prime is: ",semantic_idx_prime)

        ## get unique instance ids
        inst_orig, counts_orig = np.unique(label_orig >> 16, return_counts=True)
        # # Filter out noise and unlabeld stuff
        # index = np.where(np.isin(inst_orig ,semantic_idx_orig))
        # inst_orig = np.delete(inst_orig, index)
        # counts_orig = np.delete(counts_orig, index)
        # print(dict(zip(inst_orig, counts_orig)))
        
        inst_prime, counts_prime = np.unique(label_prime >> 16, return_counts=True)
        # # Filter out noise and unlabeled stuff
        # index = np.where(np.isin(inst_prime ,semantic_idx_orig))
        # inst_prime = np.delete(inst_prime, index)
        # counts_prime = np.delete(counts_prime, index)
        # print(dict(zip(inst_prime, counts_prime)))
        
        # for i in inst_orig:
        #     sem = np.unique(label_sem_orig[np.where(label_inst_orig == i)])
        #     if len(sem) == 0:
        #         continue
        #     if sem[0] not in semantic_classes:
        #         idx = np.where(inst_orig == i)
        #         inst_orig = np.delete(inst_orig, idx)
        #         counts_orig = np.delete(counts_orig, idx)
        # print(dict(zip(inst_orig, counts_orig)))

        # for i in inst_prime:
        #     sem = np.unique(label_sem_prime[np.where(label_inst_prime == i)])
        #     if len(sem) == 0:
        #         continue
        #     if sem[0] not in semantic_classes:
        #         idx = np.where(inst_prime == i)
        #         inst_prime = np.delete(inst_prime, idx)
        #         counts_prime = np.delete(counts_prime, idx)
        # print(dict(zip(inst_prime, counts_prime)))


        masked_prime = pc_prime[np.where(np.isin(label_inst_prime, inst_prime))]
        print("Masked Prime Shape: ", masked_prime.shape)

        for inst in inst_orig:
            print("Current Orig Instance: ", inst)
            masked_orig = pc_orig[np.where(label_inst_orig == inst)]
            print(masked_orig.shape)
            instance_list = []
            for point in tqdm(masked_orig):               
                point_scaled = np.repeat(point.reshape(-1,3), masked_prime.shape[0], axis=0)
                sqr_dist = np.linalg.norm(point_scaled - masked_prime, axis= 1)
                index = np.argmin(sqr_dist)
                value = masked_prime[index]
                real_idx = np.where(np.all(pc_prime == value, axis=1))
                instance_list.append(label_inst_prime[real_idx[0][0]])
            print(instance_list)
            c = Counter(instance_list)
            value, count = c.most_common()[0]
            iou = count / len(instance_list)
            if iou > 0.5:
                match = [inst, value]
                match_label.append(match)
                print("Match instance %d to class %d, with iou of %f" % (inst, value, iou))
                label_inst_orig[np.where(label_inst_orig == inst)] = value
            else: 
                print("Match could not reach sufficient iou with %s" % iou)
                match = [inst, inst]
                match_label.append(match)

        match_list.append(match_label)
        new_label = (label_sem_orig) + (label_inst_orig << 16)
        
        # concat_label = np.vstack((label_prime.reshape((-1,1)), new_label.reshape((-1,1))))
        # print(concat_label.shape)
        path_to_write = dst = os.path.join("./EfficientLPS/tmpDir/08/orig", os.path.basename(orig_paths[iter +1]))
        new_label.astype("int32").tofile(path_to_write)
        time.sleep(3)
    np.save("matching_table.npy", np.array(match_list))
    new_path = "_output/panoptic/sequences/08/predictions"
    try:
        os.makedirs(new_path)
    except:
        pass 
    shutil.copy("./EfficientLPS/tmpDir/08/orig", new_path)
    


if __name__ == "__main__":
    try: 
        iou_matching("./EfficientLPS/tmpDir")
    except:
        print(traceback.format_exc())