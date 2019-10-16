clc
clear

load data\test_data.mat gp1_Ydata gp2_Ydata gp1_truth gp2_truth index_truth_slice xres yres
CAPs_result_path = [pwd,'\result'];

sigma = 4/(2*log(sqrt(2)));  %8mm smooth FWHM = 4 voxel FWHM
para = struct('K_cluster_vec',2:10,'replication_in_kmeans',120,'distance_measure','corr',...
               'xres',91,'yres',109,'zres',1,'index_in_use',index_truth_slice,'per_thre',0.05,'sigma',sigma,...
               'permute_flag',1,'smooth_flag',1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% step2: obtain kmeans cluster;
[dominant_cap_map_gp1,TF_gp1,consistency_gp1,ind_t_gp1,...
    dominant_cap_map_gp2,TF_gp2,consistency_gp2,ind_t_gp2] = ...
    SL_CAP_GA_XZ(gp1_Ydata,gp2_Ydata,para,CAPs_result_path);

CORR_GP1_to_ground_truth = corr(dominant_cap_map_gp1,gp1_truth)
CORR_GP2_to_ground_truth = corr(dominant_cap_map_gp2,gp2_truth)

