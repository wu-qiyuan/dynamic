# -*- coding: utf-8 -*-
"""
Created on Wed May 11 23:58:52 2022

@author: Ding
"""
import os
import numpy as np
import cv2
import pandas as pd
import skimage.filters as ftr
from scipy import interpolate
from skimage.segmentation import active_contour
import csv
import matplotlib.pyplot as plt
import echonet

data_dir = 'E:\\R4\\test\\FinalTestData\\FinalTestData'
result_dir = 'E:\\R4\\test\\result'
def main(data_dir,result_dir,mask_model_training=False,mask_generation = False):
    print('group X domo for left verticle tracking and EF computation')
    print('data loads from :',data_dir)
    print('result is saved in:',result_dir)
    
    os.makedirs(result_dir, exist_ok=True)
    mask_dir = os.path.join(data_dir,'masks')# in batch need swich to masks
    '''step 1 mask generation'''
    if not mask_generation:
        assert os.path.isdir(mask_dir), 'mask folder does not exist, need mask generation'
    else:
        echonet.utils.segmentation.run(output="segmentation/", save_mask=True, num_workers=2)
        print(1)
    
    '''step 2 point set tracking'''
    VolumeTracing_dir = os.path.join(data_dir,'VolumeTracking.csv')
    video_folder_dir = os.path.join(data_dir,'Videos')
    mask_folder_dir = mask_dir
    result_dir_mask_mid = os.path.join(result_dir,'tracking_mask')
    
    standard_result_dir = os.path.join(result_dir,'standard_result')
    os.makedirs(standard_result_dir, exist_ok=True)
    
    tracking_frames_dir = os.path.join(data_dir,'TrackingFrames.csv')
    
    point_set_tracking(VolumeTracing_dir,video_folder_dir,mask_folder_dir,result_dir_mask_mid,standard_result_dir,tracking_frames_dir)
    
    print('point tracking is finished!')
    
    '''step 3 EF estimation'''    
    EF_result_dir = os.path.join(result_dir,'EF')
    os.makedirs(EF_result_dir, exist_ok=True)
    
    EF_computation(mask_folder_dir,EF_result_dir)
    
    print('EF estmation is finished!')
    
    print('script finished')
    
def point_set_tracking(VolumeTracing_dir,video_folder_dir,mask_folder_dir,result_dir_mask_mid,standard_result_dir,tracking_frames_dir):
    optical_flow = True
    method = 'original'  #only available when optical_flow is True  'original', 'GF_original', 'GF_mask', 'logit'
    combine = True  # only valid when optical_flow is True
    registration = True
    perform_2d = False
    perform_section = False  # only valid when registration is True
    adjust_cog = False  # only valid when registration is True
    contour_fitting = None  #'v1'  # current new method flag
#    mask_folder_dir = 'mask'
    # VolumeTracing_dir = 'SampleTestData/VolumeTracings.csv'
#    VolumeTracing_dir = 'SampleTestData/VolumeTracking.csv'
#    video_folder_dir = 'testVideo'
#    result_dir_mask_mid = 'result/mask'
#    standard_result_dir = 'result'
    standard_output = True
    opt_winsize = 15
    opt_pyramidsize = 5
    combine_weight = 0.4
    relax = 1.0
    print_optim = True

    os.makedirs(result_dir_mask_mid, exist_ok=True)
#    if standard_output:
#        os.makedirs(standard_result_dir, exist_ok=True)

    df = pd.read_csv(VolumeTracing_dir)
    mask_list = os.listdir(mask_folder_dir)
#    video_list = os.listdir(video_folder_dir)
    
    if standard_result_dir is not None:
        header = ['FileName','X1', 'Y1','Frame']
        current_standard_file_path = os.path.join(standard_result_dir, 'VolumeTracking_group4.csv')
        with open(current_standard_file_path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f,delimiter=",")
            writer.writerow(header)
        standard_result = []
    
    trackingframes = pd.read_csv(tracking_frames_dir)
    
    for mask_name in mask_list:

        current_mask_file = np.load(os.path.join(mask_folder_dir, mask_name), allow_pickle=True)
        logit = current_mask_file['arr_0']
#        logit = logit[:115,:,:]
        video_name_without_suffix_list = mask_name.split('.')  # get corresponding file name
        video_name_without_suffix_list.sort(key=lambda i: len(i), reverse=True)
        video_name_without_suffix = video_name_without_suffix_list[0]
        
        current_video_tracking_frames = trackingframes[trackingframes['FileName'].str.contains(video_name_without_suffix)]
        tracking_list = [current_video_tracking_frames['Frames_1'].max(),current_video_tracking_frames['Frames_2'].max(),current_video_tracking_frames['Frames_3'].max()]
        '''video loading'''
        original_video_cap = cv2.VideoCapture(
            os.path.join(video_folder_dir, video_name_without_suffix + '.avi'))  # video loading
        original_video = []
        while original_video_cap.isOpened():
            ret, current_img = original_video_cap.read()
            if ret:
                gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
                original_video.append(gray)
            else:
                break

        original_video = np.array(original_video)
#        original_video = original_video[:115,:,:]
        corresponding_data_points = df[df['Filename'].str.contains(video_name_without_suffix)]

        selected_frame = np.min(
            corresponding_data_points['Frame'])  # set this to 0 in final test, here just select on frame for deploying

        corresponding_data_points = corresponding_data_points[corresponding_data_points['Frame'] == selected_frame]

        '''
        OpticalFlow input pareperation:
            Given pixel collection
            mask regularization
            opticalflow params setting: windowsize, pyramid level(coarse to fine), criteria
        '''

        Initial_pos = np.array(((
            corresponding_data_points['X1'], corresponding_data_points['Y1']))).reshape(2, -1).T
        # Initial_pos = np.array(((
        #     corresponding_data_points['X1'], corresponding_data_points['X2'], corresponding_data_points['Y1'], corresponding_data_points['Y2']))).reshape(2, -1).T
        # Initial_pos[int(Initial_pos.shape[0]/2):,:] = np.flip(Initial_pos[int(Initial_pos.shape[0]/2):,:],axis=0)
        binary_mask = np.zeros_like(logit)
        binary_mask[logit > 0] = 1

        '''If the result is somewhat unchanged, pick less winSize, for unstable one pick larger one'''
        lk_params = dict(winSize=(opt_winsize, opt_winsize), maxLevel=opt_pyramidsize,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        point_set = Initial_pos. astype(np. float)
        tracking_result = np.zeros((original_video.shape[0], point_set.shape[0], 2))

        ''' Tracking: concatenate forward and backward directions'''
        '''step 1 pick the point set in the selected frame'''
        contour_points = generate_contour(num_points=400, field=binary_mask[selected_frame, :, :])
        zp = cumulativedis(contour_points)

        cluster_set = point_set_clustering(point_set.T)
        for cluster_ind in range(max(cluster_set[1,:])+1):
            cluster = point_set[cluster_set[1, :] == cluster_ind, :]
            z_1 = coord1d(cluster[0], contour_points, zp)[0] / zp[-1]
            z_2 = coord1d(cluster[-1], contour_points, zp)[0] / zp[-1]
            if cluster.shape[0] - 2.5 < 0:
                cluster_2d = np.zeros(cluster.shape)
                for i in range(cluster.shape[0]):
                    cluster_2d[i] = coord2d(coord1d(cluster[i], contour_points, zp)[0], contour_points, zp)
            else:
                mid_z = coord1d(cluster[int(cluster.shape[0] / 2)], contour_points, zp)[0] / zp[-1]
                # suppose z1 < z2
                # condition 1: z > z1 and z < z2  oppose to 0
                # condition 2: z < z1 and z < z2 or z > z1 and z > z2   with 0
                direction = (mid_z - z_1) * (mid_z - z_2)
                # if direction == 0 or z_1-z_2 == 0:
                #     raise 'Initualize projection error'

                # calc r_cluster
                r_cluster = cumulativedis(cluster) / cumulativedis(cluster)[-1]
                if direction > 0:  # with zero
                    patch_len = np.sign(z_1 - z_2) * (z_2 - z_1) + 1 + np.linalg.norm(
                        contour_points[0] - contour_points[-1]) / zp[-1]
                else:
                    patch_len = np.sign(z_1 - z_2) * (z_1 - z_2)
                z_generate = (z_1 * np.ones(r_cluster.shape) + np.sign(direction) * np.sign(
                    z_1 - z_2) * r_cluster * patch_len) % (
                                     1 + np.linalg.norm(contour_points[0] - contour_points[-1]) / zp[-1])
                cluster_2d = np.zeros(cluster.shape)
                for i in range(z_generate.shape[0]):
                    cluster_2d[i] = coord2d(z_generate[i] * zp[-1], contour_points, zp)

            point_set[cluster_set[1, :] == cluster_ind, :] = cluster_2d

        r_generated = cumulativedis(point_set) / cumulativedis(point_set)[-1]
        tracking_result[selected_frame, :, :] = Initial_pos

        '''step 2 : forward optflow'''
        print('forward processing')
        # r_generated = np.linspace(0,1,Initial_pos.shape[0])
        for current_frame in range(selected_frame, original_video.shape[0] - 1):
            print('current_frame = ')
            print(current_frame)
            point_set_update = point_set
            if optical_flow:
                if method == 'original':
                    prev_img = original_video[current_frame, :, :]
                    next_img = original_video[current_frame + 1, :, :]
                elif method == 'GF_original':
                    prev_img = ftr.gaussian(original_video[current_frame, :, :], sigma=3, preserve_range=True)
                    next_img = ftr.gaussian(original_video[current_frame + 1, :, :], sigma=3, preserve_range=True)
                elif method == 'GF_mask':
                    prev_img = 255 * ftr.gaussian(binary_mask[current_frame, :, :], sigma=1, preserve_range=True)
                    next_img = 255 * ftr.gaussian(binary_mask[current_frame + 1, :, :], sigma=1, preserve_range=True)
                elif method == 'logit':
                    prev_img = 127.5 * logit[current_frame, :, :] / np.max(logit[current_frame, :, :]) + 127.5
                    next_img = 127.5 * logit[current_frame + 1, :, :] / np.max(logit[current_frame, :, :]) + 127.5
                # elif method == 'combined':
                #     prev_img = original_video[current_frame, :, :]
                #     next_img = original_video[current_frame + 1, :, :]
                else:
                    raise ('Invalid method name.')

                prev_img = prev_img.astype(np.uint8)
                next_img = next_img.astype(np.uint8)

                point_set = point_set.astype(np.float32)

                point_set_update, st, err = cv2.calcOpticalFlowPyrLK(prev_img, next_img, point_set, None, **lk_params)

            if combine:
                # gs = 255 * ftr.gaussian(binary_mask[current_frame+1, :, :], sigma=1, preserve_range=True)
                # point_set_update = combine_method(point_set=point_set, point_set_update=point_set_update,
                #                                   field=ftr.laplace(gs, ksize=3), lmd=combine_weight)
                point_set_update = combine_method(point_set=point_set, point_set_update=point_set_update,
                                                  field=logit[current_frame + 1, :, :], lmd=combine_weight, prt=print_optim)

            if registration:

                contour_points = generate_contour(num_points=400, field=binary_mask[current_frame + 1, :, :])
                #
                # plt.plot(contour_points[:,0],contour_points[:,1])
                # plt.show()
                if perform_2d:
                    raise 'NOt Implemented'

                if perform_section:
                    for cluster_ind in range(max(cluster_set[1, :]) + 1):
                        cluster = point_set_update[cluster_set[1, :] == cluster_ind, :]
                        r_cluster = r_generated[cluster_set[1, :] == cluster_ind]
                        r_cluster = r_cluster - r_cluster[0]
                        point_set_update[cluster_set[1, :] == cluster_ind, :] = registration_method(contour_points=contour_points,
                                                               point_set_update=cluster, adjust_cog = adjust_cog, search_region = 0.5,
                                                               r_generated=r_cluster)
                else:
                    point_set_update = registration_method(contour_points=contour_points, search_region = 0.5,
                                                           point_set_update=point_set_update, r_generated=r_generated)

            if contour_fitting == 'v1':
                point_set_update = fit_contour(point_set, point_set_update, relax_coef=relax)

            tracking_result[current_frame + 1, :, :] = point_set_update
            point_set = point_set_update

        '''step 3 backwards optflow'''
        point_set = Initial_pos
        print('backward processing')
        for current_frame in range(selected_frame - 1, -1, -1):
            print('current_frame = ')
            print(current_frame)
            point_set_update = point_set
            if optical_flow:
                if method == 'original':
                    prev_img = original_video[current_frame + 1, :, :]
                    next_img = original_video[current_frame, :, :]
                elif method == 'GF_original':
                    prev_img = ftr.gaussian(original_video[current_frame + 1, :, :], sigma=3, preserve_range=True)
                    next_img = ftr.gaussian(original_video[current_frame, :, :], sigma=3, preserve_range=True)
                elif method == 'GF_mask':
                    prev_img = 255 * ftr.gaussian(binary_mask[current_frame + 1, :, :], sigma=1, preserve_range=True)
                    next_img = 255 * ftr.gaussian(binary_mask[current_frame, :, :], sigma=1, preserve_range=True)
                elif method == 'logit':
                    prev_img = 127.5 * logit[current_frame + 1, :, :] / np.max(logit[current_frame, :, :]) + 127.5
                    next_img = 127.5 * logit[current_frame, :, :] / np.max(logit[current_frame, :, :]) + 127.5
                elif method == 'combined':
                    prev_img = original_video[current_frame + 1, :, :]
                    next_img = original_video[current_frame, :, :]
                else:
                    raise ('Invalid method name.')

                prev_img = prev_img.astype(np.uint8)
                next_img = next_img.astype(np.uint8)

                point_set = point_set.astype(np.float32)

                point_set_update, st, err = cv2.calcOpticalFlowPyrLK(prev_img, next_img, point_set, None, **lk_params)


            if combine:
                # gs = 255 * ftr.gaussian(binary_mask[current_frame, :, :], sigma=1, preserve_range=True)
                # point_set_update = combine_method(point_set=point_set, point_set_update=point_set_update,
                #                                   field=ftr.laplace(gs, ksize=3), lmd=combine_weight)
                point_set_update = combine_method(point_set=point_set, point_set_update=point_set_update,
                                                  field=logit[current_frame, :, :], lmd=combine_weight, prt=print_optim)

            if registration:

                contour_points = generate_contour(num_points=400, field=binary_mask[current_frame, :, :])
                #
                # plt.plot(contour_points[:,0],contour_points[:,1])
                # plt.show()

                if perform_2d:
                    raise 'NOt Implemented'

                if perform_section:
                    for cluster_ind in range(max(cluster_set[1, :]) + 1):
                        cluster = point_set_update[cluster_set[1, :] == cluster_ind, :]
                        r_cluster = r_generated[cluster_set[1, :] == cluster_ind]
                        r_cluster = r_cluster - r_cluster[0]
                        point_set_update[cluster_set[1, :] == cluster_ind, :] = registration_method(
                            contour_points=contour_points,
                            point_set_update=cluster, adjust_cog=adjust_cog, search_region=0.5,
                            r_generated=r_cluster)
                else:
                    point_set_update = registration_method(contour_points=contour_points, search_region=0.5,
                                                           point_set_update=point_set_update, r_generated=r_generated)

            if contour_fitting == 'v1':
                point_set_update = fit_contour(point_set, point_set_update, relax_coef=relax)

            tracking_result[current_frame, :, :] = point_set_update
            point_set = point_set_update

        print('tracking of ', video_name_without_suffix, '.avi has finished.')
        np.save(os.path.join(result_dir_mask_mid, video_name_without_suffix + ('_combined_' if combine else '_') + method + '.npy'),
                tracking_result)

#        '''generating standard output for submission'''
#        if standard_output:
#            header = ['FileName','X1', 'Y1','Frame']
#            current_standard_file_path = os.path.join(standard_result_dir, video_name_without_suffix + '.csv')
#            with open(current_standard_file_path, 'w', encoding='UTF8', newline='') as f:
#                writer = csv.writer(f)
#                writer.writerow(header)
#                for i in range(tracking_result.shape[0]):
#                    current_frame_num = i
#                    for j in range(tracking_result.shape[1]):
#                        current_point_num = j
#                        current_x = tracking_result[i, j, 0]
#                        current_y = tracking_result[i, j, 1]
#                        current_line = [current_frame_num, current_point_num, current_x, current_y]
#                        writer.writerow(current_line)
        if standard_output:
            standard_result.append(tracking_result)
            np.save(os.path.join(standard_result_dir,'standard_result.npy'),standard_result)
            with open(current_standard_file_path, 'a+', encoding='UTF8', newline='') as f:
                writer = csv.writer(f,delimiter=",")
                for j in range( tracking_result.shape[0]):
                    filename = str(video_name_without_suffix)
                    filename = filename.zfill(3)+'.avi'
                    for k in range(tracking_result.shape[1]): 
                        if j in tracking_list:
                            current_line = [filename, tracking_result[j,k,0], tracking_result[j,k,1],str(j)]
                            writer.writerow(current_line)
            
#    if standard_output:
#        current_standard_file_path = os.path.join(standard_result_dir, 'VolumeTracking_group4.csv')
#        with open(current_standard_file_path, 'w', encoding='UTF8', newline='') as f:
#            writer = csv.writer(f,delimiter=",")
#            writer.writerow(header)
#            for i in range(len(standard_result)):
#                current_point_set = standard_result[i]
#                for j in range( current_point_set.shape[0]):
#                    filename = str(filename)
#                    filename = filename.zfill(3)+'.avi'
#                    for k in range(current_point_set.shape[1]):                        
#                        current_line = [filename, current_point_set[j,k,0], current_point_set[j,k,1],k]
#                        writer.writerow(current_line)
            
        

def fit_contour(  # adjusting distance between track points by relaxing moving (along edge)
        point_set,
        point_set_update,
        relax_coef = 1.0
):
    point_set_adjusted = point_set_update
    for i in range(point_set_update.shape[0]):
        edge = point_set_update[(i + 1) % point_set_update.shape[0], :] - point_set_update[
                                                                          (i - 1) % point_set_update.shape[0], :]
        vec = point_set_update[i, :] - point_set_update[(i - 1) % point_set_update.shape[0], :]
        proj = np.dot(vec, edge) / (np.linalg.norm(edge) ** 2)
        vec_old = point_set[i, :] - point_set[(i - 1) % point_set.shape[0], :]
        edge_old = point_set[(i + 1) % point_set.shape[0], :] - point_set[(i - 1) % point_set.shape[0], :]
        proj_old = np.dot(vec_old, edge_old) / (np.linalg.norm(edge_old) ** 2)
        diff = proj - proj_old
        print('diff_proj = ')
        print(diff)
        point_set_adjusted[i, :] = point_set_update[i, :] + relax_coef * diff * edge
    return point_set_adjusted


def registration_method(
        contour_points,
        point_set_update,
        r_generated,  # 1-d array of ratio
        adjust_cog = False,
        prt = False,
        perform_2d = True,
        search_region = 1.0,
        search_steps = 21
):
    zp = cumulativedis(contour_points)
    # eliminate bias of point_set_update by aligning Center of Gravity
    if adjust_cog:
        cog1 = np.sum(point_set_update, axis=0) / point_set_update.shape[0]
        cog2 = np.sum(contour_points, axis=0) / contour_points.shape[0]
        for i in range(point_set_update.shape[0]):
            point_set_update[i] = point_set_update[i] + (cog2-cog1)

    # find z of point_set_update, as ground truth for registration
    ps_update_1d = np.zeros((point_set_update.shape[0], 2))  # z and r, only first column is needed
    for i in range(point_set_update.shape[0]):
        ps_update_1d[i] = coord1d(point_set_update[i], contour_points,
                                    zp)  # project to 1-d representation (z-representation)

    # if perform_2d:
        # z_generated = (r_generated * zp[-1] + ps_update_1d[0, 0]) % (
        #         zp[-1] + np.linalg.norm(contour_points[0] - contour_points[-1]))
        # for k in np.linspace(-search_region, search_region, search_steps):
        #     pos = k * vector[i, :] + point_set_update[i, :]
        #     Loss = (1 - lmd) * np.abs(k ** 2) + lmd * np.abs(fun(pos[0], pos[1]))**2  # need interpolation
        #     if Loss < Loss_min:
        #         Loss_min = Loss
        #         k_best = k
        #         pos_best = pos
        # coord2d(z_generated[i], contour_points, zp)

    else:
        # # find z of point_set_update, as ground truth for registration
        # ps_update_1d = np.zeros((point_set_update.shape[0], 2))  # z and r, only first column is needed
        # for i in range(point_set_update.shape[0]):
        #     ps_update_1d[i] = coord1d(point_set_update[i], contour_points,
        #                               zp)  # project to 1-d representation (z-representation)

        # registration find the movement of ground truth (get offset of z and apply it on generated point set)
        # argmin (zi+dz-zupdatei)^2
        z_updated = register_z(z_infer=ps_update_1d[:, 0], z_generated=((r_generated * zp[-1] + ps_update_1d[0, 0]) % (
                zp[-1] + np.linalg.norm(contour_points[0] - contour_points[-1]))), search_region=search_region,
                               search_steps=search_steps, prt=prt)
        z_updated = register_z(z_infer=ps_update_1d[:, 0], z_generated=z_updated,
                               search_region=search_region / search_steps, search_steps=search_steps, prt=prt)


        # find coordinates from z (project back to 2-d)
        for i in range(z_updated.shape[0]):
            point_set_update[i] = coord2d(z_updated[i], contour_points, zp)


    return point_set_update


def combine_method(
        point_set,
        point_set_update,
        field,  # logit of the current/next frame (2D)
        lmd = 0.75,  # weight for logit punishment
        search_region = 1.0,
        search_steps = 21,
        prt = False
):
    vector = point_set_update - point_set
    x_grid = range(field.shape[0])
    y_grid = range(field.shape[1])
    fun = interpolate.interp2d(x_grid, y_grid, field[:, :],
                               kind='cubic')  # kind could be {'linear', 'cubic', 'quintic'}
    # vector_orth = vector
    # vector_orth[:,0] = - vector[:,1]
    # vector_orth[:,1] = vector[:,0]
    for i in range(vector.shape[0]):
        Loss_min = (1 - lmd) * np.abs(0 ** 2) + lmd * np.abs(fun(vector[i, 0], vector[i, 1]))**2
        k_best = 0
        pos_best = point_set_update[i, :]
        for k in np.linspace(-search_region, search_region, search_steps):
            pos = k * vector[i, :] + point_set_update[i, :]
            Loss = (1 - lmd) * np.abs(k ** 2) + lmd * np.abs(fun(pos[0], pos[1]))**2  # need interpolation
            if Loss < Loss_min:
                Loss_min = Loss
                k_best = k
                pos_best = pos
        for k in np.linspace(k_best-search_region/(search_steps-1), k_best+search_region/(search_steps-1), search_steps):
            pos = k * vector[i, :] + point_set_update[i, :]
            Loss = (1 - lmd) * np.abs(k ** 2) + lmd * np.abs(fun(pos[0], pos[1]))**2  # need interpolation
            if Loss < Loss_min:
                Loss_min = Loss
                k_best = k
                pos_best = pos
        if prt:
            print('Loss_min, k_best = ')
            print(Loss_min)
            print(k_best)

        point_set_update[i, :] = pos_best
    return point_set_update


def generate_contour(  # generate spline contour points
        field,
        num_points = 400,
):
    # if initualize:

        # for i in range(10):
        #     contour_point_set = combine_method(point_set=initial_point_set, point_set_update=initial_point_set,
        #                                        field=field,
        #                                        lmd=1, search_region=5, search_steps=21)
    int_points = find_contour(field)
    closed_points = np.concatenate((int_points[:, 0, :], int_points[0, 0, :].reshape((1, 2))), axis=0)
    tck, u = interpolate.splprep([closed_points[:, 0], closed_points[:, 1]], s=8)
    x_i, y_i = interpolate.splev(np.linspace(0, 1, num_points), tck)
    contour_point_set = np.concatenate((x_i.reshape((-1,1)),y_i.reshape((-1,1))),axis=1)

    # contour_point_set = combine_method(point_set=initial_point_set, point_set_update=contour_point_set, field=field,
    #                                    lmd=1, search_region=1, search_steps=101)
    return contour_point_set


def find_contour(segment):  # find integer contour from mask
    contours, hierarchy = cv2.findContours(segment.astype(int), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = cv2.contourArea(contours[0])
    cnt = contours[0]
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > max_area:
            cnt = contours[i]
            max_area = area

    cnt = cv2.approxPolyDP(cnt, 0.01, True) # * cv2.arcLength(cnt, True)
    return cnt


def register_z(
        z_infer,  # 1d array of z value, as no moving ground truth
        z_generated,  # 1d array of z value, moving part
        search_region = 1.0,
        search_steps = 21,
        prt = False,  # print or not
):
    Loss_min = np.linalg.norm(z_generated - z_infer)**2
    dz_best = 0.
    for dz in np.linspace(-search_region, search_region, search_steps):
        Loss = np.linalg.norm(z_generated + dz*np.ones(z_generated.shape) - z_infer)**2
        if Loss < Loss_min:
            Loss_min = Loss
            dz_best = dz
    if prt:
        print('Loss_min, dz_best = ')
        print(Loss_min)
        print(dz_best)

    z_generated = z_generated + dz_best*np.ones(z_generated.shape)
    return z_generated


def coord1d(p, cl, zp):
    # find the closest point
    mindis = np.linalg.norm(p - cl[0])
    index = 0
    for i in range(cl.shape[0]):
        dis = np.linalg.norm(p - cl[i])
        if dis < mindis:
            mindis = dis
            index = i

    # projection process
    q = p - cl[index]
    if index < cl.shape[0] - 1:
        n1 = (cl[index + 1] - cl[index]) / np.linalg.norm(cl[index + 1] - cl[index])
    else:
        n1 = (cl[index - 1] - cl[index]) / np.linalg.norm(cl[index - 1] - cl[index])
    if index > 0:
        n2 = (cl[index - 1] - cl[index]) / np.linalg.norm(cl[index - 1] - cl[index])
    else:
        n2 = (cl[index + 1] - cl[index]) / np.linalg.norm(cl[index + 1] - cl[index])
    if np.dot(n1, q) > np.dot(n2, q):
        n = n1
    else:
        n = -n2
    proj = np.dot(n, q)
    r = np.linalg.norm(q - proj * n)
    r_vec = (q - proj * n) / r  # normalized

    # find z coordinate
    z = zp[index] + proj

    return z, r


def coord2d(p, cl, zp):  # !! can be refined
    # find the corresponding point on centerline
    index = zp.shape[0] - 1
    for i in range(zp.shape[0]):
        if zp[i] - p > 0:
            index = i
            break
    if index > 0:
        res = p - zp[index - 1]
        n = (cl[index] - cl[index - 1]) / np.linalg.norm(cl[index] - cl[index - 1])

    else:
        res = p - zp[index]
        n = (cl[index + 1] - cl[index]) / np.linalg.norm(cl[index + 1] - cl[index])

    p_3d = cl[index - 1] + res * n
    return p_3d


def cumulativedis(cl):
    dis = np.zeros(cl.shape[0])
    dis[0] = 0.
    for i in range(1, cl.shape[0]):
        dis[i] = dis[i - 1] + np.linalg.norm(cl[i] - cl[i - 1])
    return dis


def point_set_clustering(point_set, threshold=4):
    '''
        point set,: given point set to register, 2xn_point
        threshold: decision thre for connection'''
    '''point set to connected_point_set, first row the order in point_set,second row corresponding cluster, third row start/end point'''
    connected_point_set = np.zeros((3, point_set.shape[1]))
    current_cluster = 0
    prev_position = point_set[:, 0]
    connected_point_set[2, 0] = 1  # first assume the first point is a start point of one cluster
    connected_point_set[2, point_set.shape[1] - 1] = 1
    for i in range(1, point_set.shape[1]):
        connected_point_set[0, i] = i
        current_position = point_set[:, i]
        if np.linalg.norm(prev_position - current_position) > threshold:
            current_cluster = current_cluster + 1
            connected_point_set[2, i] = 1
            connected_point_set[2, i - 1] = 1
        connected_point_set[1, i] = current_cluster
        prev_position = current_position

    if np.linalg.norm(prev_position - point_set[:,
                                      0]) <= threshold:  # if distance of first and last point less than thre, merge them
        connected_point_set[1, connected_point_set[1, :] == current_cluster] = 0
        connected_point_set[2, 0] = 0
        connected_point_set[2, point_set.shape[1] - 1] = 0
    connected_point_set = connected_point_set.astype(np.int)
    return connected_point_set

def EF_computation(mask_folder_dir,result_folder_dir):
    mask_list = os.listdir(mask_folder_dir)
    os.makedirs(result_folder_dir,exist_ok = True)

    EF_collection =[]
    for mask_name in mask_list:
        
        current_mask_file = np.load(os.path.join(mask_folder_dir,mask_name))
        current_mask = current_mask_file['arr_0']
        n_frame = current_mask.shape[0]
        
        current_mask[current_mask>=0] = 1
        current_mask[current_mask<0] = 0
        
        '''EF estimation based on given area'''
        vertricle_area = np.sum(current_mask,axis = (1,2))
        
        trim_min = sorted(vertricle_area)[round(len(vertricle_area) ** 0.05)]
        trim_max = sorted(vertricle_area)[round(len(vertricle_area) ** 0.95)]
        trim_range = trim_max - trim_min
        systole = scipy.signal.find_peaks(-vertricle_area, distance=1, prominence=(0.50 * trim_range))[0]
        diastole = scipy.signal.find_peaks(vertricle_area, distance=1, prominence=(0.50 * trim_range))[0]
        
        #distribution of systole and diastole
        peaks_sequences = -1*np.ones_like(vertricle_area)
        peaks_sequences[systole] = 0
        peaks_sequences[diastole] = 1
        
        if systole.shape[0]-diastole.shape[0]==1:
            '''check if one additional diastole is needed in the beginning or ending of peaks sequences'''
            mean_of_diastole = np.mean(vertricle_area[diastole])
            maximal_diastole_beginning = np.max(vertricle_area[:systole[0]])
            maximal_diastole_ending = np.max(vertricle_area[systole[-1]:])
            
            if maximal_diastole_beginning>0.95*mean_of_diastole and maximal_diastole_beginning>maximal_diastole_ending:
               pos = np.argmax(vertricle_area[:systole[0]])
               peaks_sequences[pos] = 1
           
            if maximal_diastole_ending>0.95*mean_of_diastole and maximal_diastole_beginning<=maximal_diastole_ending:
               pos = np.argmax(vertricle_area[systole[-1]:])+systole[-1]
               peaks_sequences[pos] = 1
         
        elif -systole.shape[0]+diastole.shape[0]==1:
            '''check if one additional systole is needed in the beginning or ending of peaks sequences'''
            mean_of_systole = np.mean(vertricle_area[systole])   
            minimal_systole_beginning = np.min(vertricle_area[:diastole[0]])
            minimal_systole_ending = np.min(vertricle_area[diastole[-1]:])
            
            if minimal_systole_beginning<1.05*mean_of_systole and minimal_systole_beginning<minimal_systole_ending:
               pos = np.argmin(vertricle_area[:diastole[0]])
               peaks_sequences[pos] = 0
           
            if minimal_systole_ending<1.05*mean_of_systole and minimal_systole_beginning>=minimal_systole_ending:
               pos = np.argmin(vertricle_area[diastole[-1]:])+diastole[-1]
               peaks_sequences[pos] = 0
               
        EF=[]
        ds_combination = np.where(peaks_sequences!=-1)[0]
        for i in range(ds_combination.shape[0]-1):
            current_systole_area = np.min((vertricle_area[ds_combination[i]],vertricle_area[ds_combination[1+i]]))
            current_diastole_area = np.max((vertricle_area[ds_combination[i]],vertricle_area[ds_combination[1+i]]))
            EF.append((current_diastole_area-current_systole_area)/current_diastole_area)
        EF=np.array(EF)
        average_EF = np.mean(EF)
        print('EF:',average_EF)
        EF_information = np.zeros((n_frame,4))
        EF_information[:,1] = np.array(range(n_frame))
        EF_information[:,2] = peaks_sequences
        EF_information[:,3] = vertricle_area
        
    #    result_struct = {'average_EF':average_EF,'EF':EF_information}
        
        video_name_without_suffix_list = mask_name.split('.')  # get corresponding file name
        video_name_without_suffix_list.sort(key=lambda i: len(i), reverse=True)
        video_name_without_suffix = video_name_without_suffix_list[0]
        EF_information[:,0] = video_name_without_suffix
    #    result_path = os.path.join(result_folder_dir,video_name_without_suffix+'.npy')
    #    np.save(result_path,result_struct)
        EF_collection.append(EF_information)
        
    header = ['FileName','Frame', 'Label','Volume']
    current_standard_file_path = os.path.join(result_folder_dir, 'Volumes_group'+str(4)+'.csv')
    with open(current_standard_file_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f,delimiter=",")
        writer.writerow(header)
        for i in range(len(EF_collection)):
            current_EF_information = EF_collection[i]
            for j in range(current_EF_information.shape[0]):
                filename = current_EF_information[j,0].astype(np.int)
                filename = filename.astype(str)
                filename = filename.zfill(3)
                current_line = [filename, current_EF_information[j,1].astype(np.int), current_EF_information[j,2].astype(np.int), current_EF_information[j,3].astype(np.int)]
                writer.writerow(current_line)

if __name__ == "__main__":
    main(data_dir,result_dir)

