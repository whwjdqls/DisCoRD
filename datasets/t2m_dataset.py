import codecs as cs
import random
import time
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

class MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split='train', debug=False,precomputed_features_dir=None):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = [] # id list comes from the split_files
        split_file = pjoin(opt.root_dir, split + '.txt')
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        if debug:
            id_list = id_list[:100]
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    # print("Motion {} is too short".format(name))
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except Exception as e:
                # Some motion may not exist in KIT dataset
                print(e)
                pass
        
        self.vqvae = None
        if precomputed_features_dir and precomputed_features_dir.split('/')[-1] == 'latents':
            self.latents = True
        else:
            self.latents = False
            
        # For VQVAE_Reconed motion
        if precomputed_features_dir:
            if self.latents:
                window_size = opt.window_size // 4
            else:
                window_size = opt.window_size
                
            self.vqvae = []
            self.vqvae_lengths = []
            for name in tqdm(id_list):
                try:
                    vqvae_motion = np.load(pjoin(precomputed_features_dir, name + '.npy'))
                    if vqvae_motion.shape[0] < window_size:
                        # print("Motion {} is too short".format(name))
                        continue
                    self.vqvae_lengths.append(vqvae_motion.shape[0] - window_size)
                    self.vqvae.append(vqvae_motion)
                except Exception as e:
                    # Some motion may not exist in KIT dataset
                    print(e)
                    pass
            self.cumsum_vqvae = np.cumsum([0] + self.vqvae_lengths)
            assert len(self.vqvae) == len(self.data), f"Length mismatch between data and vqvae: {len(self.data)} != {len(self.vqvae)}"
            
        self.cumsum = np.cumsum([0] + self.lengths)


        if split == 'train':
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias 
            ##### (08-27) NOTE #####
            # weight for the features, like the weight in the loss function for each feature
            # reference https://github.com/EricGuo5513/momask-codes/issues/61
            #########################
            
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                    joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                               joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (
                                                                          joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            # np.save(pjoin(opt.meta_dir, 'mean.npy'), mean) 
            # np.save(pjoin(opt.meta_dir, 'std.npy'), std) 
            ##### (08-27) NOTE #####
            # at trainig, save the trainig data std and mean
            # at evaluation, denormalize the predicted motion, and normalize with 
            # the evaluator's mean and std
            # reference https://github.com/EricGuo5513/momask-codes/issues/37
            #########################

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data): # denormalize the data
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # if self.vqvae is not None:
        #     if self.latents:
        #         window_size = self.opt.window_size // 4
        #     if item != 0:
        #         vqvae_id = np.searchsorted(self.cumsum_vqvae, item) - 1
        #         idx = item - self.cumsum_vqvae[vqvae_id] - 1
        #     else:
        #         vqvae_id = 0
        #         idx = 0
        #     vqvae_motion = self.vqvae[vqvae_id][idx:idx + window_size]
        #     # vqvae_motion = (vqvae_motion - self.mean) / self.std 
        #     # vqvae motions are output of the model. no need to normalize
        #     assert motion.shape[0]//4 == vqvae_motion.shape[0]
        #     return motion, vqvae_motion
        return motion
    
    
class MotionDatasetRandom(data.Dataset):
    def __init__(self, opt, mean, std, split='train', precomputed_features_dir=None):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = [] # id list comes from the split_files
        split_file = pjoin(opt.root_dir, split + '.txt')
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue

            except Exception as e:
                # Some motion may not exist in KIT dataset
                print(e)
                pass


        if split == 'train':
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias 
            ##### (08-27) NOTE #####
            # weight for the features, like the weight in the loss function for each feature
            # reference https://github.com/EricGuo5513/momask-codes/issues/61
            #########################
            
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                    joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                               joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (
                                                                          joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            # np.save(pjoin(opt.meta_dir, 'mean.npy'), mean) 
            # np.save(pjoin(opt.meta_dir, 'std.npy'), std) 
            ##### (08-27) NOTE #####
            # at trainig, save the trainig data std and mean
            # at evaluation, denormalize the predicted motion, and normalize with 
            # the evaluator's mean and std
            # reference https://github.com/EricGuo5513/momask-codes/issues/37
            #########################

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data): # denormalize the data
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, idx):
        
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion
    
    
class RefinementDataset(data.Dataset):
    def __init__(self, gt_motions, output_motions, mean, std, m_lengths=None,captions=None):
        self.gt_motions = gt_motions
        self.output_motions = output_motions
        self.captions = captions
        self.m_lengths = m_lengths
        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.gt_motions)))
        
    def __len__(self):
        return len(self.gt_motions)

    def __getitem__(self, idx):
        if self.m_lengths is not None:
            m_length = self.m_lengths[idx]
            # padding_mask = np.concatenate([np.ones(m_length), np.zeros(196 - m_length)], axis=0)
            padding_mask = torch.nn.functional.pad(torch.zeros(m_length), (0, 196 - m_length), value=1).bool()
            output_motion = self.output_motions[idx][:m_length]
            padded_motion = torch.nn.functional.pad(output_motion, (0,0,0, 196 - m_length), value=0)
            return self.gt_motions[idx], padded_motion, m_length, self.captions[idx], padding_mask
        else:
            return self.gt_motions[idx], self.output_motions[idx]
        
    def inv_transform(self, data): # denormalize the data
        return data * self.std + self.mean
    
    
class RFDecoderDataset(data.Dataset):
    def __init__(self, gt_motions, z, mean, std, m_lengths=None,captions=None):
        self.gt_motions = gt_motions
        self.z = z
        self.captions = captions
        self.m_lengths = m_lengths
        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.gt_motions)))
        
    def __len__(self):
        return len(self.gt_motions)

    def __getitem__(self, idx):
        if self.m_lengths is not None:
            m_length = self.m_lengths[idx]
            # padding_mask = np.concatenate([np.ones(m_length), np.zeros(196 - m_length)], axis=0)
            padding_mask = torch.nn.functional.pad(torch.zeros(m_length), (0, 196 - m_length), value=1).bool()
            z = self.z[idx]
            z_length = z.shape[0]
            padded_z = torch.nn.functional.pad(z, (0,0,0, 49 - z_length), value=0)
            # padded_motion = torch.nn.functional.pad(output_motion, (0,0,0, 196 - m_length), value=0)
            return self.gt_motions[idx], padded_z, m_length, self.captions[idx], padding_mask
        else:
            return self.gt_motions[idx], self.z[idx]
        
    def inv_transform(self, data): # denormalize the data
        return data * self.std + self.mean
    
    
class RefinementDatasetEval(data.Dataset):
    def __init__(self, gt_motions, output_motions, mean, std, word_embeddings, pos_one_hots, captions, sent_lens, m_lengths, tokens):
        self.gt_motions = gt_motions
        self.output_motions = output_motions
        self.word_embeddings = word_embeddings
        self.pos_one_hots = pos_one_hots
        self.captions = captions
        self.sent_lens = sent_lens
        self.captions = captions
        self.m_lengths = m_lengths
        self.tokens = tokens
        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.gt_motions)))
        
    def __len__(self):
        return len(self.gt_motions)

    def __getitem__(self, idx):
        return self.word_embeddings[idx], self.pos_one_hots[idx], self.captions[idx], self.sent_lens[idx], self.gt_motions[idx], self.output_motions[idx], self.m_lengths[idx], self.tokens[idx]
        
    def inv_transform(self, data): # denormalize the data
        return data * self.std + self.mean

def make_refinement_dataset(dataset, vqvae, batch_size):
    # Precompute the VQ-VAE outputs for the entire dataset using batch inference
    print("Precomputing VQ-VAE outputs for the entire dataset...")
    start_time = time.time()
    assert vqvae.__class__.__name__ in ['MotionPriorWrapper', 'VQVAE_251']
    gt_motions, output_motions = [], []
    if dataset.__class__.__name__ == 'Text2MotionDataset':
        m_lengths, captions = [], []
    else:
        m_lengths, captions = None, None
        
    device = next(vqvae.parameters()).device
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in tqdm(data_loader):
        if dataset.__class__.__name__ == 'Text2MotionDataset':
            caption, motion, m_length = batch
            m_lengths.extend(m_length)
            captions.extend(caption)
        elif dataset.__class__.__name__ == 'MotionDataset':
            motion = batch
            
        gt_batch = motion.float().to(device)
        with torch.no_grad():
            output_batch = vqvae(gt_batch.to(device))[0].to(device)

        gt_motions.extend(gt_batch.cpu())
        output_motions.extend(output_batch.cpu())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Precomputing completed in {elapsed_time:.2f} seconds.")
    # Create an instance of the RefinementDataset with the precomputed motions
    refinement_dataset = RefinementDataset(gt_motions, output_motions, dataset.mean, dataset.std, m_lengths=m_lengths, captions=captions)
    # get rid of vqvae from gpu
    vqvae.cpu()
    return refinement_dataset


def make_rf_decoder_dataset(dataset, vqvae, batch_size):
    # Precompute the VQ-VAE encodings (i.e from the codebooks) for the entire dataset using batch inference
    print("Precomputing VQ-VAE outputs for the entire dataset...")
    start_time = time.time()
    assert vqvae.__class__.__name__ in ['MotionPriorWrapper', 'VQVAE_251']
    gt_motions, output_encodings = [], []
    if dataset.__class__.__name__ == 'Text2MotionDataset': # this is when we use full length dataset
        m_lengths, captions = [], []
    else:
        m_lengths, captions = None, None
        
    device = next(vqvae.parameters()).device
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in tqdm(data_loader):
        if dataset.__class__.__name__ == 'Text2MotionDataset':
            caption, motion, m_length = batch
            m_lengths.extend(m_length)
            captions.extend(caption)
        elif dataset.__class__.__name__ == 'MotionDataset':
            motion = batch
            
        gt_batch = motion.float().to(device)
        with torch.no_grad():
            output_batch = vqvae.get_z(gt_batch.to(device)).to(device)
        # output_batch =  torch.nn.functional.interpolate(output_batch, scale_factor=4, mode='nearest')
        output_batch = output_batch.permute(0, 2, 1)
        # (B, encoder dim, seq_len/q) -> (B, encoder_dim, seq_len)
        gt_motions.extend(gt_batch.cpu())
        output_encodings.extend(output_batch.cpu())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Precomputing completed in {elapsed_time:.2f} seconds.")
    # Create an instance of the RefinementDataset with the precomputed motions
    refinement_dataset = RFDecoderDataset(gt_motions, output_encodings, dataset.mean, dataset.std, m_lengths=m_lengths, captions=captions)
    # get rid of vqvae from gpu
    vqvae.cpu()
    return refinement_dataset

def make_refinement_dataset_eval(dataset, vqvae, batch_size):
    # Precompute the VQ-VAE outputs for the entire dataset using batch inference
    print("Precomputing VQ-VAE outputs for the entire dataset...")
    start_time = time.time()
    assert vqvae.__class__.__name__ in ['MotionPriorWrapper', 'VQVAE_251']
    gt_motions, output_motions = [], []
    if dataset.__class__.__name__ == 'Text2MotionDatasetEval':
        word_embeddings, pos_one_hots, captions, sent_lens, m_lengths, tokens = [], [], [], [], [], []
    else:
        raise NotImplementedError(f"Dataset class {dataset.__class__.__name__} not supported.")
        
    device = next(vqvae.parameters()).device
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in tqdm(data_loader):
        if dataset.__class__.__name__ == 'Text2MotionDatasetEval':
            word_embedding, pos_one_hot, caption, sent_len, motion, m_length, token = batch
            word_embeddings.extend(word_embedding)
            pos_one_hots.extend(pos_one_hot)
            captions.extend(caption)
            sent_lens.extend(sent_len)
            m_lengths.extend(m_length)
            tokens.extend(token)
            
        gt_batch = motion.float().to(device)
        with torch.no_grad():
            output_batch = vqvae(gt_batch.to(device))[0].to(device)

        gt_motions.extend(gt_batch.cpu())
        output_motions.extend(output_batch.cpu())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Precomputing completed in {elapsed_time:.2f} seconds.")
    # Create an instance of the RefinementDataset with the precomputed motions
    refinement_dataset = RefinementDatasetEval(gt_motions, output_motions, dataset.mean, dataset.std, word_embeddings, pos_one_hots, captions, sent_lens, m_lengths, tokens)
    vqvae.cpu()
    return refinement_dataset

class FullSequenceMotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split='train'):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        id_list = [] # id list comes from the split_files
        self.max_motion_length = opt.max_motion_length
        min_motion_length = opt.min_motion_length
        split_file = pjoin(opt.root_dir, split + '.txt')
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < min_motion_length:
                    # print("Motion {} is too short".format(name))
                    continue
                if self.max_motion_length < motion.shape[0]:
                    motion = motion[motion.shape[0]-self.max_motion_length:]
                self.data.append(motion)
            except Exception as e:
                # Some motion may not exist in KIT dataset
                print(e)
                pass
        
        self.vqvae = None
        # For VQVAE_Reconed motion
        if hasattr(opt, 'vqvae_motion_dir'):
            self.vqvae = []
            for name in tqdm(id_list):
                try:
                    vqvae_motion = np.load(pjoin(opt.vqvae_motion_dir, name + '.npy'))
                    if vqvae_motion.shape[0] < min_motion_length:
                        # print("Motion {} is too short".format(name))
                        continue
                    if self.max_motion_length < vqvae_motion.shape[0]:
                        vqvae_motion = vqvae_motion[vqvae_motion.shape[0]-self.max_motion_length:]
                    self.vqvae.append(vqvae_motion)
                except Exception as e:
                    # Some motion may not exist in KIT dataset
                    print(e)
                    pass

        if split == 'train':
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias 
            ##### (08-27) NOTE #####
            # weight for the features, like the weight in the loss function for each feature
            # reference https://github.com/EricGuo5513/momask-codes/issues/61
            #########################
            
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                    joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                               joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                                4 + (
                                                                          joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            # np.save(pjoin(opt.meta_dir, 'mean.npy'), mean) 
            # np.save(pjoin(opt.meta_dir, 'std.npy'), std) 
            ##### (08-27) NOTE #####
            # at trainig, save the trainig data std and mean
            # at evaluation, denormalize the predicted motion, and normalize with 
            # the evaluator's mean and std
            # reference https://github.com/EricGuo5513/momask-codes/issues/37
            #########################

        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data): # denormalize the data
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]
        "Z Normalization"
        motion = (motion - self.mean) / self.std
        
        m_length = motion.shape[0]
        mask = np.concatenate([np.ones(m_length), np.zeros(self.max_motion_length - m_length)], axis=0)
        
        if motion.shape[0] < self.max_motion_length:
            motion = np.concatenate([motion,
                                    np.zeros((self.max_motion_length - motion.shape[0], motion.shape[1]))
                                    ], axis=0)

        motion = torch.from_numpy(motion).float()
        if self.vqvae is not None:
            vqvae_motion = self.vqvae[item]
            vqvae_motion = (vqvae_motion - self.mean) / self.std
            
            if vqvae_motion.shape[0] < self.max_motion_length:
                vqvae_motion = np.concatenate([vqvae_motion,
                                                np.zeros((self.max_motion_length - vqvae_motion.shape[0], vqvae_motion.shape[1]))
                                                ], axis=0)
            vqvae_motion = torch.from_numpy(vqvae_motion).float()
            return motion, vqvae_motion, m_length, mask
        return motion, m_length, mask

class Text2MotionDatasetEval(data.Dataset):
    def __init__(self, opt, mean, std, w_vectorizer, split='test', debug=False, return_old_name=False, get_first_text=False):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = 196 # just fix this bevause it is same for kit and humanml3d
        min_motion_len = opt.min_motion_length ## 40 for t2m, 24 for kit
        self.return_old_name = return_old_name
        self.get_first_text=get_first_text
        data_dict = {}
        id_list = []

        split_file = pjoin(opt.root_dir, split + '.txt')
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # data = np.load(opt.root_dir +'human_motion_data.npz', allow_pickle=True)
        # id_list = data[split].item().keys()

        new_name_list = []
        length_list = []

        if debug:
            id_list = id_list[:100]
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                # motion = data[split].item()[name]
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    # print(f"Motion {name} is too short or too long")
                    continue
                text_data = []
                flag = False
                # loading text data
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                        'length': len(n_motion),
                                                        'text':[text_dict],
                                                        'original_name': name}
                               
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'text': text_data,
                                        'original_name': name}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                print(f"Error in loading motion data for {name}")
                pass
        name_list, length_list= zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean # this is from the training data of the model that is being evaluated
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list, original_name = data['motion'], data['length'], data['text'], data['original_name']
        # Randomly select a caption
        if self.get_first_text:
            text_data = text_list[0]
        else:
            text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        motion = torch.from_numpy(motion).float()
        if self.return_old_name:
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), original_name
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split='train'):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = 196 # fixed as it is same for kit and humanml3d
        min_motion_len = opt.min_motion_length ## 40 for t2m, 24 for kit

        data_dict = {}
        id_list = []
        if split == 'debug':
            is_debug = True
        else:
            is_debug = False
        if is_debug:
            split = 'train' # use train and only use the first 1000 data
            
        # data = np.load(opt.root_dir +'human_motion_data.npz', allow_pickle=True)
        # id_list = data[split].item().keys()
        split_file = pjoin(opt.root_dir, split + '.txt')
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        if is_debug:
            id_list = id_list[:1000]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                # motion = data[split].item()[name]
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    # print(f"Motion {name} is too short or too long")
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        # print(line)
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except Exception as e:
                print(e)
                pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        name_list, length_list = new_name_list, length_list

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        motion = torch.from_numpy(motion).float() 
        return caption, motion, m_length

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        
        
# if __name__ == '__main__':
    