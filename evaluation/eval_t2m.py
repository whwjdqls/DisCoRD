import os
import sys

import clip
import numpy as np
import torch
import torch.nn.functional as F
from MotionPriors import MotionPrior

from evaluation.get_opt import get_opt

# from scipy import linalg
from evaluation.metrics import *

# import visualization.plot_3d_global as plot_3d
from evaluation.motion_process import recover_from_ric
from evaluation.t2m_eval_wrapper import EvaluatorModelWrapper
from scipy.ndimage import gaussian_filter


def calculate_area(jerk, gt_jerk):
    seq_len = jerk.shape[0]
    epsilon = 1e-6
    static_area = np.sum(np.maximum(0, gt_jerk - jerk) / (np.absolute(gt_jerk) + np.absolute(jerk) + epsilon))
    noise_area = np.sum(np.maximum(0, jerk - gt_jerk) / (np.absolute(gt_jerk) + np.absolute(jerk) + epsilon))
    static_area = static_area / seq_len
    noise_area = noise_area / seq_len
    return static_area, noise_area

# def calculate_area_v2(jerk, gt_jerk):
#     static_area = np.sum(np.maximum(0, gt_jerk - jerk))/ np.sum(np.absolute(gt_jerk) + np.absolute(jerk))  # Area where jerk is less than GT jerk (red area)
#     noise_area = np.sum(np.maximum(0, jerk - gt_jerk))/np.sum(np.absolute(gt_jerk) + np.absolute(jerk))  # Area where jerk is greater than GT jerk (blue area)
#     return static_area, noise_area
def calculate_area_v2(jerk, gt_jerk):
    static_area = np.sum(np.maximum(0, gt_jerk - jerk))/ np.sum(gt_jerk + jerk)  # Area where jerk is less than GT jerk (red area)
    noise_area = np.sum(np.maximum(0, jerk - gt_jerk))/np.sum(gt_jerk + jerk)  # Area where jerk is greater than GT jerk (blue area)
    return static_area, noise_area
#
# def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
#     xyz = xyz[:1]
#     bs, seq = xyz.shape[:2]
#     xyz = xyz.reshape(bs, seq, -1, 3)
#     plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(), title_batch, outname)
#     plot_xyz = np.transpose(plot_xyz, (0, 1, 4, 2, 3))
#     writer.add_video(tag, plot_xyz, nb_iter, fps=20)


@torch.no_grad()
def evaluation_vqvae(
    out_dir,
    val_loader,
    net,
    writer,
    ep,
    best_fid,
    best_div,
    best_top1,
    best_top2,
    best_top3,
    best_matching,
    eval_wrapper,
    save=True,
    draw=True,
):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_commit, perplexity = net(motion)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length
        )

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        "--> \t Eva. Ep %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f"
        % (
            ep,
            fid,
            diversity_real,
            diversity,
            R_precision_real[0],
            R_precision_real[1],
            R_precision_real[2],
            R_precision[0],
            R_precision[1],
            R_precision[2],
            matching_score_real,
            matching_score_pred,
        )
    )
    # logger.info(msg)fplus_j
    print(msg)

    if draw:
        writer.add_scalar("./Test/FID", fid, ep)
        writer.add_scalar("./Test/Diversity", diversity, ep)
        writer.add_scalar("./Test/top1", R_precision[0], ep)
        writer.add_scalar("./Test/top2", R_precision[1], ep)
        writer.add_scalar("./Test/top3", R_precision[2], ep)
        writer.add_scalar("./Test/matching_score", matching_score_pred, ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw:
            print(msg)
        best_fid = fid
        if save:
            torch.save({"vq_model": net.state_dict(), "ep": ep}, os.path.join(out_dir, "net_best_fid.tar"))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!" % (best_div, diversity)
        if draw:
            print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw:
            print(msg)
        best_top1 = R_precision[0]
        # if save:
        #     torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw:
            print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw:
            print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw:
            print(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({"vq_model": net.state_dict(), "ep": ep}, os.path.join(out_dir, "net_best_mm.tar"))

    # if save:
    #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer

@torch.no_grad()
def evaluation_vqvae_plus_mpjpe_on_white_noise(val_loader,noise_std, repeat_id, eval_wrapper, num_joint, test_mean, test_std, mean, std):
    # net.eval()
    # device = next(net.parameters()).device
    device = torch.device("cuda")
    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    pred_static_area = 0
    pred_noise_area = 0
    pred_static_area_v2 = 0
    pred_noise_area_v2 = 0
    pred_jerk=0
    gt_jerk=0
    num_poses = 0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch
        motion = motion.to(device)
        bs, seq = motion.shape[0], motion.shape[1]
        text_embedding = None
        # pre_pose_eval = motion + torch.randn_like(motion) * noise_std

        motion = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        # pred_pose_eval = val_loader.dastaset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        pred_pose_eval = (torch.randn_like(torch.tensor(motion)).numpy() * noise_std) + mean
        
        motion = (motion - test_mean) / test_std
        pred_pose_eval = (pred_pose_eval - test_mean) / test_std
        
        motion = torch.from_numpy(motion).to(device)
        pred_pose_eval = torch.from_numpy(pred_pose_eval).to(device)
        
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        bgt = motion.detach().cpu().numpy() * test_std + test_mean
        bpred = pred_pose_eval.detach().cpu().numpy() * test_std + test_mean
        # bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        # bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        
        
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, : m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, : m_length[i]]).float(), num_joint)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            # print(calculate_mpjpe(gt, pred).shape, gt.shape, pred.shape)
            num_poses += gt.shape[0]
            
            if pred.shape[1] == 22:
                pred = pred[:,[10,11,20,21],:]
                gt = gt[:,[10,11,20,21],:]
                fps = 20
            elif pred.shape[1] == 21:
                pred = pred[:,[7,10],:]
                gt = gt[:,[7,10],:]
                # pred = pred[:,[15,20,7,10],:]
                # gt = gt[:,[15,20,7,10],:]
                fps = 12.5
                pred = pred/1000
                gt = gt/1000
            pred_sample_jerk = calculate_finite_difference_jerk(pred, fps=fps)
            gt_sample_jerk = calculate_finite_difference_jerk(gt, fps=fps)
            pred_jerk += pred_sample_jerk.mean()
            gt_jerk += gt_sample_jerk.mean()

            static_area, noise_area = calculate_area(pred_sample_jerk, gt_sample_jerk)
            pred_static_area += static_area
            pred_noise_area += noise_area

            static_area, noise_area = calculate_area_v2(pred_sample_jerk, gt_sample_jerk)
            pred_static_area_v2 += static_area
            pred_noise_area_v2 += noise_area
        # print(mpjpe, num_poses)
        # exit()

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)


    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses
    
    pred_static_area = pred_static_area / nb_sample
    pred_noise_area = pred_noise_area / nb_sample
    pred_static_area_v2 = pred_static_area_v2 / nb_sample
    pred_noise_area_v2 = pred_noise_area_v2 / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    # Format message with added metrics
    msg = (
        "--> \t Eva. Re %d:, FID. %.4f, GT Jerk %.4f, Pred Jerk %.4f, Diversity Real. %.4f, Diversity. %.4f, "
        "R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, "
        "matching_pred. %.4f, MPJPE. %.4f, Static Area. %.4f, Noise Area. %.4f, Static Area V2. %.4f, Noise Area V2. %.4f"
        % (
            repeat_id,
            fid,
            gt_jerk,
            pred_jerk,
            diversity_real,
            diversity,
            R_precision_real[0],
            R_precision_real[1],
            R_precision_real[2],
            R_precision[0],
            R_precision[1],
            R_precision[2],
            matching_score_real,
            matching_score_pred,
            mpjpe,
            pred_static_area,
            pred_noise_area,
            pred_static_area_v2,
            pred_noise_area_v2,
        )
    )

    # Log or print message
    # logger.info(msg)
    print(msg)
    sys.stdout.flush()

    # Return values, including added metrics if needed
    return fid, diversity, R_precision, matching_score_pred, mpjpe, gt_jerk, pred_jerk, pred_static_area, pred_noise_area, pred_static_area_v2, pred_noise_area_v2


@torch.no_grad()
def evaluation_vqvae_plus_mpjpe_on_noisy_data(val_loader,noise_std, repeat_id, eval_wrapper, num_joint, test_mean, test_std, data_type="noise"):
    # net.eval()
    # device = next(net.parameters()).device
    device = torch.device("cuda")
    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    pred_static_area = 0
    pred_noise_area = 0
    pred_static_area_v2 = 0
    pred_noise_area_v2 = 0
    pred_jerk=0
    gt_jerk=0
    num_poses = 0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch
        motion = motion.to(device)
        bs, seq = motion.shape[0], motion.shape[1]
        text_embedding = None
        # pre_pose_eval = motion + torch.randn_like(motion) * noise_std

        motion = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        # pred_pose_eval = val_loader.dastaset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        if data_type == "smooth":
            pred_pose_eval = gaussian_filter(motion, sigma=(0, noise_std, 0))+ np.random.randn(*motion.shape) * 0.00001
        elif data_type == "noise":
            pred_pose_eval = motion + np.random.randn(*motion.shape) * noise_std
        elif data_type == "smooth_noise":
            pred_pose_eval = gaussian_filter(motion, sigma=(0, noise_std, 0)) + np.random.randn(*motion.shape) * noise_std
        
        motion = (motion - test_mean) / test_std
        pred_pose_eval = (pred_pose_eval - test_mean) / test_std
        
        motion = torch.from_numpy(motion).to(device)
        pred_pose_eval = torch.from_numpy(pred_pose_eval).to(device)
        
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        bgt = motion.detach().cpu().numpy() * test_std + test_mean
        bpred = pred_pose_eval.detach().cpu().numpy() * test_std + test_mean
        # bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        # bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        
        
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, : m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, : m_length[i]]).float(), num_joint)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            # print(calculate_mpjpe(gt, pred).shape, gt.shape, pred.shape)
            num_poses += gt.shape[0]
            
            if pred.shape[1] == 22:
                pred = pred[:,[10,11,20,21],:]
                gt = gt[:,[10,11,20,21],:]
                fps = 20
            elif pred.shape[1] == 21:
                pred = pred[:,[7,10],:]
                gt = gt[:,[7,10],:]
                # pred = pred[:,[15,20,7,10],:]
                # gt = gt[:,[15,20,7,10],:]
                fps = 12.5
                pred = pred/1000
                gt = gt/1000
            pred_sample_jerk = calculate_finite_difference_jerk(pred, fps=fps)
            gt_sample_jerk = calculate_finite_difference_jerk(gt, fps=fps)
            pred_jerk += pred_sample_jerk.mean()
            gt_jerk += gt_sample_jerk.mean()

            static_area, noise_area = calculate_area(pred_sample_jerk, gt_sample_jerk)
            pred_static_area += static_area
            pred_noise_area += noise_area

            static_area, noise_area = calculate_area_v2(pred_sample_jerk, gt_sample_jerk)
            pred_static_area_v2 += static_area
            pred_noise_area_v2 += noise_area
        # print(mpjpe, num_poses)
        # exit()

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)


    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses
    
    pred_static_area = pred_static_area / nb_sample
    pred_noise_area = pred_noise_area / nb_sample
    pred_static_area_v2 = pred_static_area_v2 / nb_sample
    pred_noise_area_v2 = pred_noise_area_v2 / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    # Format message with added metrics
    msg = (
        "--> \t Eva. Re %d:, FID. %.4f, GT Jerk %.4f, Pred Jerk %.4f, Diversity Real. %.4f, Diversity. %.4f, "
        "R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, "
        "matching_pred. %.4f, MPJPE. %.4f, Static Area. %.4f, Noise Area. %.4f, Static Area V2. %.4f, Noise Area V2. %.4f"
        % (
            repeat_id,
            fid,
            gt_jerk,
            pred_jerk,
            diversity_real,
            diversity,
            R_precision_real[0],
            R_precision_real[1],
            R_precision_real[2],
            R_precision[0],
            R_precision[1],
            R_precision[2],
            matching_score_real,
            matching_score_pred,
            mpjpe,
            pred_static_area,
            pred_noise_area,
            pred_static_area_v2,
            pred_noise_area_v2,
        )
    )

    # Log or print message
    # logger.info(msg)
    print(msg)
    sys.stdout.flush()

    # Return values, including added metrics if needed
    return fid, diversity, R_precision, matching_score_pred, mpjpe, gt_jerk, pred_jerk, pred_static_area, pred_noise_area, pred_static_area_v2, pred_noise_area_v2


# Step 7: Calculate Jerk Using Finite Differences
def calculate_finite_difference_jerk(joints, fps=20):
    jerk = (joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3]) * (fps ** 3)
    jerk_magnitude = np.linalg.norm(jerk, axis=-1)
    return jerk_magnitude

@torch.no_grad()
def evaluation_vqvae_plus_mpjpe_plus_jerk(val_loader, net, repeat_id, eval_wrapper, num_joint, test_mean, test_std, normalize_with_test=True,smooth_sigma=0.0):
    net.eval()
    device = next(net.parameters()).device
    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    num_poses = 0
    pred_static_area = 0
    pred_noise_area = 0
    pred_static_area_v2 = 0
    pred_noise_area_v2 = 0
    pred_jerk=0
    gt_jerk=0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch
        motion = motion.to(device)
        bs, seq = motion.shape[0], motion.shape[1]
        text_embedding = None
        if net.__class__.__name__ == "MotionPriorWrapper":
            if 'text_condition' in net.model_cfg.model.keys() and net.model_cfg.model.text_condition:
                text_embedding = net.model.net.encode_text(caption) 
            if 'full_motion' in net.model_cfg.train.keys() and net.model_cfg.train.full_motion:
                pred_pose_eval,others = net(motion, text_embedding=text_embedding, m_length=m_length)
            else: 
                pred_pose_eval,others = net(motion, text_embedding=text_embedding, m_length=None)
            
        elif net.__class__.__name__ == "RVQVAE":
            pred_pose_eval, commit_loss, perplexity = net(motion)
        else:
            try:
                pred_pose_eval, mu, logvar = net(motion)
            except:
                raise ValueError("Unknown model type")
        # denormalize with training data
        motion = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        pred_pose_eval = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        
        if smooth_sigma > 0:
            pred_pose_eval = gaussian_filter(pred_pose_eval, sigma=(0, smooth_sigma, 0)) # smooth along the time axis
            
        if normalize_with_test: # noramlixe with test data
            motion = (motion - test_mean) / test_std
            pred_pose_eval = (pred_pose_eval - test_mean) / test_std
            
        motion = torch.from_numpy(motion).to(device)
        pred_pose_eval = torch.from_numpy(pred_pose_eval).to(device)
        
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)
        if normalize_with_test:
            bgt = motion.detach().cpu().numpy() * test_std + test_mean
            bpred = pred_pose_eval.detach().cpu().numpy() * test_std + test_mean
        else:
            bgt = motion.detach().cpu().numpy()
            bpred = pred_pose_eval.detach().cpu().numpy()
        # bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        # bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        
        
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, : m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, : m_length[i]]).float(), num_joint)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            
            if pred.shape[1] == 22:
                pred = pred[:,[10,11,20,21],:]
                gt = gt[:,[10,11,20,21],:]
                fps = 20
            elif pred.shape[1] == 21:
                pred = pred[:,[7,10],:]
                gt = gt[:,[7,10],:]
                # pred = pred[:,[15,20,7,10],:]
                # gt = gt[:,[15,20,7,10],:]
                fps = 12.5
                pred = pred/1000
                gt = gt/1000
            pred_sample_jerk = calculate_finite_difference_jerk(pred, fps=fps)
            gt_sample_jerk = calculate_finite_difference_jerk(gt, fps=fps)
            pred_jerk += pred_sample_jerk.mean()
            gt_jerk += gt_sample_jerk.mean()

            static_area, noise_area = calculate_area(pred_sample_jerk, gt_sample_jerk)
            pred_static_area += static_area
            pred_noise_area += noise_area

            static_area, noise_area = calculate_area_v2(pred_sample_jerk, gt_sample_jerk)
            pred_static_area_v2 += static_area
            pred_noise_area_v2 += noise_area

            # print(calculate_mpjpe(gt, pred).shape, gt.shape, pred.shape)
            num_poses += gt.shape[0]

        # print(mpjpe, num_poses)
        # exit()

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)


    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses
    gt_jerk = gt_jerk / nb_sample
    pred_jerk = pred_jerk / nb_sample
    pred_static_area = pred_static_area / nb_sample
    pred_noise_area = pred_noise_area / nb_sample
    pred_static_area_v2 = pred_static_area_v2 / nb_sample
    pred_noise_area_v2 = pred_noise_area_v2 / nb_sample
    
    
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    # Format message with added metrics
    msg = (
        "--> \t Eva. Re %d:, FID. %.4f, GT Jerk %.4f, Pred Jerk %.4f, Diversity Real. %.4f, Diversity. %.4f, "
        "R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, "
        "matching_pred. %.4f, MPJPE. %.4f, Static Area. %.4f, Noise Area. %.4f, Static Area V2. %.4f, Noise Area V2. %.4f"
        % (
            repeat_id,
            fid,
            gt_jerk,
            pred_jerk,
            diversity_real,
            diversity,
            R_precision_real[0],
            R_precision_real[1],
            R_precision_real[2],
            R_precision[0],
            R_precision[1],
            R_precision[2],
            matching_score_real,
            matching_score_pred,
            mpjpe,
            pred_static_area,
            pred_noise_area,
            pred_static_area_v2,
            pred_noise_area_v2,
        )
    )

    # Log or print message
    # logger.info(msg)
    print(msg)
    sys.stdout.flush()

    # Return values, including added metrics if needed
    return fid, diversity, R_precision, matching_score_pred, mpjpe, gt_jerk, pred_jerk, pred_static_area, pred_noise_area, pred_static_area_v2, pred_noise_area_v2

@torch.no_grad()
def evaluation_vqvae_plus_mpjpe(val_loader, net, repeat_id, eval_wrapper, num_joint, test_mean, test_std, smooth_sigma=0.0):
    net.eval()
    device = next(net.parameters()).device
    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    num_poses = 0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch
        motion = motion.to(device)
        bs, seq = motion.shape[0], motion.shape[1]
        text_embedding = None
        if net.__class__.__name__ == "MotionPriorWrapper":
            if 'text_condition' in net.model_cfg.model.keys() and net.model_cfg.model.text_condition:
                text_embedding = net.model.net.encode_text(caption) 
            if 'full_motion' in net.model_cfg.train.keys() and net.model_cfg.train.full_motion:
                pred_pose_eval,others = net(motion, text_embedding=text_embedding, m_length=m_length)
            else: 
                pred_pose_eval,others = net(motion, text_embedding=text_embedding, m_length=None)
            
        elif net.__class__.__name__ == "RVQVAE":
            pred_pose_eval, commit_loss, perplexity = net(motion)
        else:
            try:
                pred_pose_eval, mu, logvar = net(motion)
            except:
                raise ValueError("Unknown model type")
            
        motion = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        pred_pose_eval = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        
        if smooth_sigma > 0:
            pred_pose_eval = gaussian_filter(pred_pose_eval, sigma=(0, smooth_sigma, 0)) # smooth along the time axis
        
        motion = (motion - test_mean) / test_std
        pred_pose_eval = (pred_pose_eval - test_mean) / test_std
        
        motion = torch.from_numpy(motion).to(device)
        pred_pose_eval = torch.from_numpy(pred_pose_eval).to(device)
        
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        bgt = motion.detach().cpu().numpy() * test_std + test_mean
        bpred = pred_pose_eval.detach().cpu().numpy() * test_std + test_mean
        # bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        # bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        
        
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, : m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, : m_length[i]]).float(), num_joint)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            # print(calculate_mpjpe(gt, pred).shape, gt.shape, pred.shape)
            num_poses += gt.shape[0]

        # print(mpjpe, num_poses)
        # exit()

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)


    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, MPJPE. %.4f"
        % (
            repeat_id,
            fid,
            diversity_real,
            diversity,
            R_precision_real[0],
            R_precision_real[1],
            R_precision_real[2],
            R_precision[0],
            R_precision[1],
            R_precision[2],
            matching_score_real,
            matching_score_pred,
            mpjpe,
        )
    )
    # logger.info(msg)
    print(msg)
    sys.stdout.flush()
    return fid, diversity, R_precision, matching_score_pred, mpjpe


@torch.no_grad()
def evaluation_vqvae_plus_l1(val_loader, net, repeat_id, eval_wrapper, num_joint):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    l1_dist = 0
    num_poses = 1
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_commit, perplexity = net(motion)
        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length
        )

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, : m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, : m_length[i]]).float(), num_joint)
            # gt = motion[i, :m_length[i]]
            # pred = pred_pose_eval[i, :m_length[i]]
            num_pose = gt.shape[0]
            l1_dist += F.l1_loss(gt, pred) * num_pose
            num_poses += num_pose

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f"
        % (
            repeat_id,
            fid,
            diversity_real,
            diversity,
            R_precision_real[0],
            R_precision_real[1],
            R_precision_real[2],
            R_precision[0],
            R_precision[1],
            R_precision[2],
            matching_score_real,
            matching_score_pred,
            l1_dist,
        )
    )
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist


@torch.no_grad()
def evaluation_res_plus_l1(val_loader, vq_model, res_model, repeat_id, eval_wrapper, num_joint, do_vq_res=True):
    vq_model.eval()
    res_model.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    l1_dist = 0
    num_poses = 1
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        if do_vq_res:
            code_ids, all_codes = vq_model.encode(motion)
            if len(code_ids.shape) == 3:
                pred_vq_codes = res_model(code_ids[..., 0])
            else:
                pred_vq_codes = res_model(code_ids)
            # pred_vq_codes = pred_vq_codes - pred_vq_res + all_codes[1:].sum(0)
            pred_pose_eval = vq_model.decoder(pred_vq_codes)
        else:
            rec_motions, _, _ = vq_model(motion)
            pred_pose_eval = res_model(rec_motions)  # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length
        )

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, : m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, : m_length[i]]).float(), num_joint)
            # gt = motion[i, :m_length[i]]
            # pred = pred_pose_eval[i, :m_length[i]]
            num_pose = gt.shape[0]
            l1_dist += F.l1_loss(gt, pred) * num_pose
            num_poses += num_pose

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f"
        % (
            repeat_id,
            fid,
            diversity_real,
            diversity,
            R_precision_real[0],
            R_precision_real[1],
            R_precision_real[2],
            R_precision[0],
            R_precision[1],
            R_precision[2],
            matching_score_real,
            matching_score_pred,
            l1_dist,
        )
    )
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist


@torch.no_grad()
def evaluation_mask_transformer(
    out_dir,
    val_loader,
    trans,
    vq_model,
    writer,
    ep,
    best_fid,
    best_div,
    best_top1,
    best_top2,
    best_top3,
    best_matching,
    eval_wrapper,
    plot_func,
    save_ckpt=False,
    save_anim=False,
):
    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith("clip_model.")]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            "t2m_transformer": t2m_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            "ep": ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    time_steps = 18
    if "kit" in out_dir:
        cond_scale = 2
    else:
        cond_scale = 4

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # (b, seqlen)
        mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale, temperature=1)

        # motion_codes = motion_codes.permute(0, 2, 1)
        mids.unsqueeze_(-1)
        pred_motions = vq_model.forward_decoder(mids)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length
        )

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    writer.add_scalar("./Test/FID", fid, ep)
    writer.add_scalar("./Test/Diversity", diversity, ep)
    writer.add_scalar("./Test/top1", R_precision[0], ep)
    writer.add_scalar("./Test/top2", R_precision[1], ep)
    writer.add_scalar("./Test/top3", R_precision[2], ep)
    writer.add_scalar("./Test/matching_score", matching_score_pred, ep)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, "model", "net_best_fid.tar"), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, "animation", "E%04d" % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer


@torch.no_grad()
def evaluation_res_transformer(
    out_dir,
    val_loader,
    trans,
    vq_model,
    writer,
    ep,
    best_fid,
    best_div,
    best_top1,
    best_top2,
    best_top3,
    best_matching,
    eval_wrapper,
    plot_func,
    save_ckpt=False,
    save_anim=False,
    cond_scale=2,
    temperature=1,
):
    def save(file_name, ep):
        res_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith("clip_model.")]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            "res_transformer": res_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            "ep": ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda().long()
        pose = pose.cuda().float()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        code_indices, all_codes = vq_model.encode(pose)
        # (b, seqlen)
        if ep == 0:
            pred_ids = code_indices[..., 0:1]
        else:
            pred_ids = trans.generate(
                code_indices[..., 0], clip_text, m_length // 4, temperature=temperature, cond_scale=cond_scale
            )
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length
        )

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    writer.add_scalar("./Test/FID", fid, ep)
    writer.add_scalar("./Test/Diversity", diversity, ep)
    writer.add_scalar("./Test/top1", R_precision[0], ep)
    writer.add_scalar("./Test/top2", R_precision[1], ep)
    writer.add_scalar("./Test/top3", R_precision[2], ep)
    writer.add_scalar("./Test/matching_score", matching_score_pred, ep)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, "model", "net_best_fid.tar"), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, "animation", "E%04d" % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer


@torch.no_grad()
def evaluation_res_transformer_plus_l1(
    val_loader,
    vq_model,
    trans,
    repeat_id,
    eval_wrapper,
    num_joint,
    cond_scale=2,
    temperature=1,
    topkr=0.9,
    cal_l1=True,
):
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    l1_dist = 0
    num_poses = 1
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda().long()
        pose = pose.cuda().float()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        code_indices, all_codes = vq_model.encode(pose)
        # print(code_indices[0:2, :, 1])

        pred_ids = trans.generate(
            code_indices[..., 0],
            clip_text,
            m_length // 4,
            topk_filter_thres=topkr,
            temperature=temperature,
            cond_scale=cond_scale,
        )
        # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids)

        if cal_l1:
            bgt = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
            bpred = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy())
            for i in range(bs):
                gt = recover_from_ric(torch.from_numpy(bgt[i, : m_length[i]]).float(), num_joint)
                pred = recover_from_ric(torch.from_numpy(bpred[i, : m_length[i]]).float(), num_joint)
                # gt = motion[i, :m_length[i]]
                # pred = pred_pose_eval[i, :m_length[i]]
                num_pose = gt.shape[0]
                l1_dist += F.l1_loss(gt, pred) * num_pose
                num_poses += num_pose

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length
        )

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f"
        % (
            repeat_id,
            fid,
            diversity_real,
            diversity,
            R_precision_real[0],
            R_precision_real[1],
            R_precision_real[2],
            R_precision[0],
            R_precision[1],
            R_precision[2],
            matching_score_real,
            matching_score_pred,
            l1_dist,
        )
    )
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist


@torch.no_grad()
def evaluation_mask_transformer_test(
    val_loader,
    vq_model,
    trans,
    repeat_id,
    eval_wrapper,
    time_steps,
    cond_scale,
    temperature,
    topkr,
    gsample=True,
    force_mask=False,
    cal_mm=True,
):
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(val_loader):
        # print(i)
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
            # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(
                    clip_text,
                    m_length // 4,
                    time_steps,
                    cond_scale,
                    temperature=temperature,
                    topk_filter_thres=topkr,
                    gsample=gsample,
                    force_mask=force_mask,
                )

                # motion_codes = motion_codes.permute(0, 2, 1)
                mids.unsqueeze_(-1)
                pred_motions = vq_model.forward_decoder(mids)

                et_pred, em_pred = eval_wrapper.get_co_embeddings(
                    word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length
                )
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1)  # (bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(
                clip_text,
                m_length // 4,
                time_steps,
                cond_scale,
                temperature=temperature,
                topk_filter_thres=topkr,
                force_mask=force_mask,
            )

            # motion_codes = motion_codes.permute(0, 2, 1)
            mids.unsqueeze_(-1)
            pred_motions = vq_model.forward_decoder(mids)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length
            )

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, "
        f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "
        f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, "
        f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f},"
        f"multimodality. {multimodality:.4f}"
    )
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality


@torch.no_grad()
def evaluation_mask_transformer_test_plus_res(
    val_loader,
    vq_model,
    res_model,
    trans,
    repeat_id,
    eval_wrapper,
    time_steps,
    cond_scale,
    temperature,
    topkr,
    rf_model=None,
    gsample=True,
    force_mask=False,
    cal_mm=True,
    res_cond_scale=5,
):
    trans.eval()
    vq_model.eval()
    res_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3
# use tqdm with enumerate
    # for i, batch in enumerate(tqdm.tqdm(val_loader)):
    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        if i < num_mm_batch:
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(
                    clip_text,
                    m_length // 4,
                    time_steps,
                    cond_scale,
                    temperature=temperature,
                    topk_filter_thres=topkr,
                    gsample=gsample,
                    force_mask=force_mask,
                )

                pred_ids = res_model.generate(mids, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
                pred_motions = vq_model.forward_decoder(pred_ids)
                # if pred_motions.shape[1] != 196:
                #     print('pred_motions shape', pred_motions.shape)
                #     print('pred_ids shape', pred_ids.shape)
                #     # print('m_length', m_length)
                #     # print("padding pred_motions and pred_ids in evaluation_mask_transformer_test_plus_res")
                #     pred_motions = torch.cat([pred_motions, torch.zeros(pred_motions.shape[0], 196 - pred_motions.shape[1], pred_motions.shape[2]).cuda()], dim=1)
                #     pred_ids = torch.cat([pred_ids, torch.zeros(pred_ids.shape[0], 49 - pred_ids.shape[1], pred_ids.shape[2]).cuda()], dim=1)
                if rf_model is not None:
                    if 'text_condition' in rf_model.model_cfg.model.keys() and rf_model.model_cfg.model.text_condition:
                        text_embedding = rf_model.model.net.encode_text(clip_text) 
                    else:
                        text_embedding = None
                        
                    if rf_model.model_cfg.model.name == "RFfromNoise":
                        if 'full_motion' in rf_model.model_cfg.train.keys() and rf_model.model_cfg.train.full_motion:
                            pred_motions = rf_model.refine_from_Noise(pred_motions, text_embedding=text_embedding, m_length=m_length)
                        else:
                            pred_motions = rf_model.refine_from_Noise(pred_motions, text_embedding=text_embedding, m_length=None)
                            
                    if rf_model.model.__class__.__name__ == 'Flow':
                        if 'full_motion' in rf_model.model_cfg.train.keys() and rf_model.model_cfg.train.full_motion:
                            pred_motions = rf_model.refine(pred_motions, pred_ids=pred_ids, text_embedding=text_embedding, m_length=m_length)
                        else: 
                            pred_motions = rf_model.refine(pred_motions, pred_ids=pred_ids, text_embedding=text_embedding, m_length=None)
                            
                    elif rf_model.model.__class__.__name__ == 'RectifiedFlowDecoder':
                        if 'full_motion' in rf_model.model_cfg.train.keys() and rf_model.model_cfg.train.full_motion:
                            pred_motions = rf_model.decode_with_RF(pred_ids, text_embedding=text_embedding, m_length=m_length)
                        else: 
                            pred_motions = rf_model.decode_with_RF(pred_ids, text_embedding=text_embedding, m_length=None)
                        
                et_pred, em_pred = eval_wrapper.get_co_embeddings(
                    word_embeddings, pos_one_hots, sent_len,  pred_motions.clone(), m_length
                )
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1)  # (bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(
                clip_text,
                m_length // 4,
                time_steps,
                cond_scale,
                temperature=temperature,
                topk_filter_thres=topkr,
                force_mask=force_mask,
            )
            pred_ids = res_model.generate(mids, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
            pred_motions = vq_model.forward_decoder(pred_ids)
            # if pred_motions.shape[1] != 196:
            #     print('pred_motions shape', pred_motions.shape)
            #     print('pred_ids shape', pred_ids.shape)
            #     print('m_length', m_length)
            #     # pad
            #     print("padding pred_motions and pred_ids in evaluation_mask_transformer_test_plus_res")
            #     pred_motions = torch.cat([pred_motions, torch.zeros(pred_motions.shape[0], 196 - pred_motions.shape[1], pred_motions.shape[2]).cuda()], dim=1)
            #     pred_ids = torch.cat([pred_ids, torch.zeros(pred_ids.shape[0], 49 - pred_ids.shape[1], pred_ids.shape[2]).cuda()], dim=1)
        # pred_motions = vq_model.forward_decoder(mids)
            if rf_model is not None:
                if 'text_condition' in rf_model.model_cfg.model.keys() and rf_model.model_cfg.model.text_condition:
                    text_embedding = rf_model.model.net.encode_text(clip_text) 
                else:
                    text_embedding = None
                
                if rf_model.model_cfg.model.name == "RFfromNoise":
                    if 'full_motion' in rf_model.model_cfg.train.keys() and rf_model.model_cfg.train.full_motion:
                        pred_motions = rf_model.refine_from_Noise(pred_motions, text_embedding=text_embedding, m_length=m_length)
                    else:
                        pred_motions = rf_model.refine_from_Noise(pred_motions, text_embedding=text_embedding, m_length=None)
                        
                if rf_model.model.__class__.__name__ == 'Flow':
                    if 'full_motion' in rf_model.model_cfg.train.keys() and rf_model.model_cfg.train.full_motion:
                        pred_motions = rf_model.refine(pred_motions, pred_ids=pred_ids, text_embedding=text_embedding, m_length=m_length)
                    else: 
                        pred_motions = rf_model.refine(pred_motions, pred_ids=pred_ids, text_embedding=text_embedding, m_length=None)
                        
                elif rf_model.model.__class__.__name__ == 'RectifiedFlowDecoder':
                    if 'full_motion' in rf_model.model_cfg.train.keys() and rf_model.model_cfg.train.full_motion:
                        pred_motions = rf_model.decode_with_RF(pred_ids, text_embedding=text_embedding, m_length=m_length)
                    else: 
                        pred_motions = rf_model.decode_with_RF(pred_ids, text_embedding=text_embedding, m_length=None)
                
            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len,  pred_motions.clone(), m_length
            )

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,  pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, "
        f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "
        f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, "
        f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f},"
        f"multimodality. {multimodality:.4f}"
    )
    print(msg)
    sys.stdout.flush()
    return fid, diversity, R_precision, matching_score_pred, multimodality


@torch.no_grad()
def evaluation_marm(
    val_loader,
    vae_model,
    marm,
    repeat_id,
    eval_wrapper,
    cfg,
    cfg_schedule,
    temperature,
    force_mask=False,
    cal_mm=True,
    use_gt_length=False,
    save_dir_path=None,
):
    # 09-10 NOTE
    # i'm not sure what is force_mask yet..let's exclude it for now
    save_flag = False
    if (save_dir_path is not None) and repeat_id == 0:  # only save for the first repeat when the save_path is provided
        os.makedirs(save_dir_path, exist_ok=True)
        save_flag = True

    vae_model.eval()
    marm.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
            # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                ## original code from momask ##
                # mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
                #                       temperature=temperature, topk_filter_thres=topkr,
                #                       gsample=gsample, force_mask=force_mask)
                # pred_ids = res_model.generate(mids, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
                # pred_motions = vq_model.forward_decoder(pred_ids)
                #################################

                texts = clip.tokenize(clip_text, truncate=True).cuda()

                if use_gt_length:
                    # convert torch tensor into int
                    max_seq_len = m_length.item() // 4  # quant factor
                    num_iter = max_seq_len // 4  # heuristic
                else:
                    raise NotImplementedError

                z = marm.generate(
                    bsz=bs,
                    num_iter=num_iter,
                    max_seq_len=m_length,
                    cfg=cfg,
                    cfg_schedule=cfg_schedule,
                    texts=texts,
                    temperature=temperature,
                    progress=True,
                )
                pred_motions = vae_model.decode(z)

                et_pred, em_pred = eval_wrapper.get_co_embeddings(
                    word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length
                )
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1)  # (bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            # mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
            #                       temperature=temperature, topk_filter_thres=topkr,
            #                       force_mask=force_mask)
            # pred_ids = res_model.generate(mids, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
            # pred_motions = vq_model.forward_decoder(pred_ids)
            pred_motions_batch = []
            for i, text in enumerate(clip_text):
                text_tokenized = clip.tokenize(text, truncate=True).cuda()
                if use_gt_length:
                    max_seq_len = m_length[i].item() // 4
                    num_iter = max_seq_len // 4
                else:
                    raise NotImplementedError
                z = marm.generate(
                    bsz=1,
                    num_iter=num_iter,
                    max_seq_len=max_seq_len,
                    cfg=cfg,
                    cfg_schedule=cfg_schedule,
                    texts=text_tokenized,
                    temperature=temperature,
                    progress=True,
                )
                pred_motions = vae_model.decode(z)

                if save_flag:
                    save_motions = pred_motions.cpu().numpy()
                    save_file_path = os.path.join(save_dir_path, text + ".npy")
                    np.save(save_file_path, save_motions)

                pred_m_length = pred_motions.shape[1]
                if pred_m_length < 196:
                    padding = torch.zeros(1, 196 - pred_m_length, pred_motions.shape[2], device=pred_motions.device)
                    pred_motions = torch.cat([pred_motions, padding], dim=1)
                pred_motions_batch.append(pred_motions)

            pred_motions_batch = torch.cat(pred_motions_batch, dim=0)
            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_motions_batch.clone(), m_length
            )

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (
        f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, "
        f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "
        f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, "
        f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f},"
        f"multimodality. {multimodality:.4f}"
    )
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality


def evaluate_motion_prior(test_dataloader, vae_model, model_cfg, device, vqvae=None, train_data="t2m", repeat_time=1):
    # this part is from momask-codes
    if train_data == "t2m":
        test_mean = np.load("./datasets/t2m-mean.npy")
        test_std = np.load("./datasets/t2m-std.npy")
    elif train_data == "kit":
        test_mean = np.load("./datasets/kit_mean.npy")
        test_std = np.load("./datasets/kit_std.npy")
    
    dataset_opt_path = f"evaluation/models/{train_data}/Comp_v6_KLD005/opt.txt"
    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    EvaluatorModelWrapper.device = device
    nb_joints = 21 if train_data == "kit" else 22

    vae_model.eval()
    vae_model.to(device)

    net = MotionPrior.MotionPriorWrapper(model_cfg=model_cfg, model_ckpt=None, device=device)
    net.model = vae_model

    if vqvae is not None:
        net.vqvae = vqvae
    net.eval()
    net.to(device)
    
    fid, div, top1, top2, top3, matching, mae = [], [], [], [], [], [], []

    for i in range(repeat_time):
        best_fid, best_div, Rprecision, best_matching, l1_dist = evaluation_vqvae_plus_mpjpe(
            test_dataloader, net, i, eval_wrapper=eval_wrapper, num_joint=nb_joints, test_mean=test_mean, test_std=test_std
        )
        fid.append(best_fid)
        div.append(best_div)
        top1.append(Rprecision[0])
        top2.append(Rprecision[1])
        top3.append(Rprecision[2])
        matching.append(best_matching)
        mae.append(l1_dist)

    fid = np.mean(np.array(fid))
    div = np.mean(np.array(div))
    top1 = np.mean(np.array(top1))
    top2 = np.mean(np.array(top2))
    top3 = np.mean(np.array(top3))
    matching = np.mean(np.array(matching))
    mae = np.mean(np.array(mae))
    metric_dict = {
        "FID": fid,
        "Diversity": div,
        "Top1": top1,
        "Top2": top2,
        "Top3": top3,
        "Matching": matching,
        "MAE": mae,
    }
    return metric_dict


import tqdm
@torch.no_grad()        
def evaluation_transformer_t2m(out_dir, val_loader, net, trans, nb_iter, clip_model, eval_wrapper, rf_model=None, save = False, savegif=False, savenpy=False) : 
    trans.eval()
    nb_sample = 0

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    num_mm_batch = 1
    # for batch in tqdm.tqdm(val_loader):
    for j, batch in tqdm.tqdm(enumerate(val_loader)):

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        
        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text = clip_model.encode_text(text).float()
        
        
        if j < num_mm_batch:
            motion_multimodality_batch = []
            for i in range(11):
                print("iter", i)
                pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
                pred_len = torch.ones(bs).long()
                
                for k in range(bs):
                    try:
                        index_motion = trans.sample(feat_clip_text[k:k+1], True)
                    except:
                        print('Error in sampling')
                        index_motion = torch.ones(1,1).cuda().long()

                    pred_pose = net.forward_decoder(index_motion)
                    cur_len = pred_pose.shape[1]

                    pred_len[k] = min(cur_len, seq)
                    pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
                    

                    # if savenpy:
                    #     pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    #     pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    #     if savenpy:
                    #         np.save(os.path.join(out_dir, name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

                # pred_pose_eval = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

                motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))
        else:
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()
            
            for k in range(bs):
                try:
                    index_motion = trans.sample(feat_clip_text[k:k+1], True)
                except:
                    print('Error in sampling')
                    index_motion = torch.ones(1,1).cuda().long()

                pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if i == 0 and savenpy:
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if savenpy:
                        np.save(os.path.join(out_dir, name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

            pose = pose.cuda().float()
            
            et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
            motion_annotation_list.append(em)
            motion_pred_list.append(em_pred)

            if savenpy:
                pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                if savenpy:
                    for j in range(bs):
                        np.save(os.path.join(out_dir, name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())


            # temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
            temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
            temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
            R_precision_real += temp_R
            matching_score_real += temp_match
            # temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
            temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
            temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
            R_precision += temp_R
            matching_score_pred += temp_match

            nb_sample += bs

        

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality