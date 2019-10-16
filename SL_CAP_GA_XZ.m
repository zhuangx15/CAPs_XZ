%% main function
function [dominant_cap_map_gp1,TF_gp1,consistency_gp1,ind_t_gp1,...
    dominant_cap_map_gp2,TF_gp2,consistency_gp2,ind_t_gp2] = ...
    SL_CAP_GA_XZ(Ydata_gp1,Ydata_gp2,para,CAPs_result_path)
%======================================================================================
%%% Contact zhuangx@ccf.org for bugs or questions 
%  All rights reserved.
%
%  Reference Paper: 
%=====================================================================================
    %% computes spatially less overlapping dominant-CAP sets for each group 
    % computes temporal fraction, spatial consistency and swithing
    % probability of each network. 
    %%%% inputs: 
    %    Ydata_gp1 and Ydata_gp2: network-associated time frames;  time-frames whose seed signal intensity is among the top);
    %                             tdim*q;
    %    para: K_cluster_vec: number of clusters in each kmeans run;
    %          replication_in_kmeans: repeat each kmeans; 
    %          distance_measure: 'corr' for 1-correlation as distance measure;
    %          xres: 91; yres 109; zres: 91; for MNI 152 2mm template;
    %          index_in_use: voxel_index_within_mask; MNI152-2mm template is used here; index of voxels within the template brain; can be replaced by a brain mask;
    %          per_thre: p-value threshold of the null-distribution in determining whether current d-CAP candidate is spatially similar to existing d-CAPs.
    %          sigma: standard deviation of the Gaussian filter applied to smooth the fMRI time series;
    %          permute flag: 1;
    %          smooth flag: 1;
    %          example: para = struct('K_cluster_vec',K_cluster_vec,'replication_in_kmeans',120,'distance_measure','corr',...
    %                                 'xres',91,'yres',109,'zres',91,'index_in_use',index_in_use,'per_thre',0.05,'sigma',sigma,...
    %                                 'permute_flag',1,'smooth_flag',1);
    %%%% outputs: 
    %     ind_t_gp1/gp2: reassignment of each time point to the d-CAPs; used to compute switching probability;
    %     dominant_cap_map_gp1/gp2: d-CAP in each group; 
    %     TF_gp1/gp2: temporal fraction of each d-CAP in each group;
    %     consistency_gp1/gp2: spatial consistency of each d-CAP in each group;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%;
    %% step1: kmeans clustering with multiple ks;
    % kmeans run use parallel automatically; 
    K_cluster_vec = para.K_cluster_vec;
    replication_in_kmeans = para.replication_in_kmeans;
    K_means_type = para.distance_measure;
    N_K_cluster_vec = length(K_cluster_vec);
    for j = 1:N_K_cluster_vec
        K_cluster = K_cluster_vec(j);
        kmeans_save_path = [CAPs_result_path,'\CAP_result_',num2str(K_cluster),'_clusters'];
        if ~exist(kmeans_save_path,'dir')
            mkdir(kmeans_save_path);
        end
        [id_t] = function_clustering_both_groups...
            (Ydata_gp1,Ydata_gp2,K_cluster,replication_in_kmeans,K_means_type);
        save([kmeans_save_path,'\matrices'],'id_t');
    end
    %% step2: determine d-CAPs
    initial_gp1 = mean(Ydata_gp1);
    initial_gp2 = mean(Ydata_gp2);  % initilization from entire group average;
    [dominant_cap_map_gp1,dominant_cap_map_gp2,dominant_CAPs_path_gp1,dominant_CAPs_path_gp2] = determine_compute_dCAP_set_each_group...
    (Ydata_gp1,Ydata_gp2,initial_gp1,initial_gp2,CAPs_result_path,CAPs_result_path,para);
    
    %% step3: quantatitive measurements;
    [new_avg_dominant_caps_gp1,new_avg_dominant_caps_zscore_gp1,TF_gp1,consistency_gp1,ind_t_gp1,max_corr_t_gp1] = ...
    caps_group_analysis_quantative_measurement_calculation(dominant_cap_map_gp1,Ydata_gp1);
    [new_avg_dominant_caps_gp2,new_avg_dominant_caps_zscore_gp2,TF_gp2,consistency_gp2,ind_t_gp2,max_corr_t_gp2] = ...
    caps_group_analysis_quantative_measurement_calculation(dominant_cap_map_gp2,Ydata_gp2);
end

%% functions perform clustering
function [id_t] = function_clustering_both_groups...
    (Ydata_gp1,Ydata_gp2,K_cluster,replicated_K_means,K_means_type)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%           
% Output: id_t{1}: cluster assignment for group1
%         id_t{2}: cluster assignment for group2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nt_gp1 = size(Ydata_gp1,1); Nt_gp2 = size(Ydata_gp2,1);
Ydata_t = [Ydata_gp1;Ydata_gp2];
if strcmp(K_means_type,'corr') == 1
    [id] = kmeans(Ydata_t,K_cluster,'replicate',replicated_K_means,'Distance','correlation','Options',statset('UseParallel',1));
elseif strcmp(K_means_type,'Euclidean') == 1
    [id] = kmeans(Ydata_t,K_cluster,'replicate',replicated_K_means,'Options',statset('UseParallel',1));
end
id_gp1 = id(1:Nt_gp1,:);
id_gp2 = id(Nt_gp1+1:end);
id_t = cell(2,1);
id_t{1} = id_gp1;
id_t{2} = id_gp2;
end

%% functions determine d-CAPs
function [dominant_cap_map_gp1,dominant_cap_map_gp2,dominant_CAPs_path_gp1,dominant_CAPs_path_gp2] = determine_compute_dCAP_set_each_group...
    (Ydata_gp1,Ydata_gp2,initial_gp1,initial_gp2,kmeans_result_path,dCAPs_result_path,para)
    
    kmeans_cluster = para.K_cluster_vec;
    tdim_gp1 = size(Ydata_gp1,1);
    dominant_cap_map_gp1(:,1) = initial_gp1;
    thre_dominant_caps_gp1{1,1} = [];
    thre_corr_between_dCAPs_cluster_gp1{1,1} = [];
    dominant_cap_name_gp1 = {'group average'};
    
    dominant_cap_map_gp2(:,1) = initial_gp2;
    thre_dominant_caps_gp2{1,1} = [];
    thre_corr_between_dCAPs_cluster_gp2{1,1} = [];
    dominant_cap_name_gp2 = {'group average'};

    N_K = length(kmeans_cluster);
    for i = 1:N_K
        K = kmeans_cluster(i);
        para.K_cluster = K;
        CAPs_result_K_path = [kmeans_result_path,'\CAP_result_',num2str(K),'_clusters'];
        load([CAPs_result_K_path,'\matrices.mat'],'id_t');
        id_t = [id_t{1,1};id_t{2,1}];
        id_t = sort_id(id_t,K);
        id_gp1 = id_t(1:tdim_gp1,:);
        id_gp2 = id_t(tdim_gp1+1:end,:);
        for kid = 1:K
            para.k = kid;
            [cluster_avg_map_gp1,cluster_avg_map_gp2,index_gp1,index_gp2] = function_kmeans_group_specific_map(id_gp1,id_gp2,Ydata_gp1,Ydata_gp2,kid);
            if sum(index_gp1) == 0 && sum(index_gp2) == 0
                continue;
            end
            if sum(index_gp1) == 0
                [dominant_cap_map_gp2,dominant_cap_name_gp2,thre_dominant_caps_gp2,thre_corr_between_dCAPs_cluster_gp2] = function_dCAPs_set_determination...
                    (cluster_avg_map_gp2,dominant_cap_map_gp2,dominant_cap_name_gp2,thre_dominant_caps_gp2,thre_corr_between_dCAPs_cluster_gp2,para);
                continue;
            end
            if sum(index_gp2) == 0
                [dominant_cap_map_gp1,dominant_cap_name_gp1,thre_dominant_caps_gp1,thre_corr_between_dCAPs_cluster_gp1] = function_dCAPs_set_determination...
                    (cluster_avg_map_gp1,dominant_cap_map_gp1,dominant_cap_name_gp1,thre_dominant_caps_gp1,thre_corr_between_dCAPs_cluster_gp1,para);
                continue;
            end
            [dominant_cap_map_gp2,dominant_cap_name_gp2,thre_dominant_caps_gp2,thre_corr_between_dCAPs_cluster_gp2] = function_dCAPs_set_determination...
                (cluster_avg_map_gp2,dominant_cap_map_gp2,dominant_cap_name_gp2,thre_dominant_caps_gp2,thre_corr_between_dCAPs_cluster_gp2,para);
            [dominant_cap_map_gp1,dominant_cap_name_gp1,thre_dominant_caps_gp1,thre_corr_between_dCAPs_cluster_gp1] = function_dCAPs_set_determination...
                (cluster_avg_map_gp1,dominant_cap_map_gp1,dominant_cap_name_gp1,thre_dominant_caps_gp1,thre_corr_between_dCAPs_cluster_gp1,para);
        end
    end
    count_dominant_caps_gp1 = size(dominant_cap_map_gp1,2);
    count_dominant_caps_gp2 = size(dominant_cap_map_gp2,2);
    dominant_CAPs_path_gp1 = [dCAPs_result_path,'\GP1\',num2str(count_dominant_caps_gp1),'dominantCAPs'];
    if ~exist(dominant_CAPs_path_gp1,'dir')
        mkdir(dominant_CAPs_path_gp1);
    end
    save([dominant_CAPs_path_gp1,'\dominant_CAPs_matrix.mat'],'dominant_cap_map_gp1','dominant_cap_name_gp1',...
        'thre_corr_between_dCAPs_cluster_gp1','thre_dominant_caps_gp1');
    
    dominant_CAPs_path_gp2 = [dCAPs_result_path,'\GP2\',num2str(count_dominant_caps_gp2),'dominantCAPs'];
    if ~exist(dominant_CAPs_path_gp2,'dir')
        mkdir(dominant_CAPs_path_gp2);
    end
    save([dominant_CAPs_path_gp2,'\dominant_CAPs_matrix.mat'],'dominant_cap_map_gp2','dominant_cap_name_gp2',...
        'thre_corr_between_dCAPs_cluster_gp2','thre_dominant_caps_gp1');
end


function [dominant_cap_set,dominant_cap_name,thre_dominant_caps,thre_corr_between_dCAPs_cluster] = ...
    function_dCAPs_set_determination(cluster_avg_map,dominant_cap_set,dominant_cap_name,thre_dominant_caps,thre_corr_between_dCAPs_cluster,para)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    per_thre = para.per_thre;
    index_in_use = para.index_in_use; xres = para.xres; yres = para.yres; zres = para.zres;
    sigma = para.sigma;
    permute_flag = 1;
    smooth_flag = 1;
    K_cluster = para.K_cluster;
    k_current = para.k;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [~,count_dominant_caps] = size(dominant_cap_set);
    corr_between_cluster = zeros(count_dominant_caps,1);
    thre = ones(count_dominant_caps,1);
    for k = 1:count_dominant_caps
        corr_between_cluster(k,1) = corr(cluster_avg_map(:),dominant_cap_set(:,k));
        [thre(k,1),corr_null] = corr_permutation(cluster_avg_map(:),dominant_cap_set(:,k),per_thre,index_in_use,...
            xres,yres,zres,5,sigma,1000,permute_flag,smooth_flag,1);
    end
    index = ((corr_between_cluster < thre ));
    if sum(index) == length(index)
        index1 = find((corr_between_cluster<0.01) && (abs(corr_between_cluster-thre)<0.01));  % control for random noise
        if isempty(index1)
            count_dominant_caps = count_dominant_caps + 1;
            thre_dominant_caps{count_dominant_caps,1} = thre;
            thre_corr_between_dCAPs_cluster{count_dominant_caps,1} = corr_between_cluster;
            dominant_cap_set(:,count_dominant_caps) = cluster_avg_map(:);
            dominant_cap_name{count_dominant_caps,1} = [num2str(K_cluster),'cluster_CAPs_',num2str(k_current)];
        end
    end
end


function [cluster_avg_gp1_map,cluster_avg_gp2_map,index_gp1_k,index_gp2_k] = function_kmeans_group_specific_map...
    (id_gp1,id_gp2,Ydata_gp1_reduced,Ydata_gp2_reduced,k)
index_gp1_k = (id_gp1 == k);
index_gp2_k = (id_gp2 == k);
if sum(index_gp1_k) == 0
    cluster_avg_gp1_map = [];
elseif sum(index_gp1_k) == 1
    cluster_avg_gp1_map = Ydata_gp1_reduced(index_gp1_k,:);
else
    cluster_avg_gp1_map = mean(Ydata_gp1_reduced(index_gp1_k,:));
end

if sum(index_gp2_k) == 0
     cluster_avg_gp2_map = [];
elseif sum(index_gp2_k) == 1
    cluster_avg_gp2_map = Ydata_gp2_reduced(index_gp2_k,:);
else
    cluster_avg_gp2_map = mean(Ydata_gp2_reduced(index_gp2_k,:));
end
end

function [id_t_sort,TF_t_sort] = sort_id(id_t,K)
    TF_t = zeros(K,1);
    for i = 1:K   % calculate TF, start from max TF in K=20;
        TF_t(i,1) = sum(id_t==i)/length(id_t);
    end
    [TF_t_sort,ind_TF_t_sort] = sort(TF_t,'descend');
    id_t_sort = zeros(size(id_t));
    for i = 1:K
        index = id_t == ind_TF_t_sort(i);
        id_t_sort(index,1) = i;
    end
end

function [thre,corr_null] = corr_permutation...
    (group1_map,group2_map,per_thre,index_MNI2mm,xres,yres,zres,window_size,sigma,N_iter,permute_flag,smooth_flag,use_parallel)
    if nargin <13
       use_parallel = 0;
    end
    corr_null = zeros(N_iter,1);
%     tic
    if use_parallel == 1
        parfor j = 1:N_iter
            corr_null(j,1) = corr_permutation_single(group1_map,group2_map,index_MNI2mm,xres,yres,zres,window_size,sigma,permute_flag,smooth_flag); 
        end
    else   
        for j = 1:N_iter
            corr_null(j,1) = corr_permutation_single(group1_map,group2_map,index_MNI2mm,xres,yres,zres,window_size,sigma,permute_flag,smooth_flag); 
        end
    end
%     toc
    [yy,xx] = ecdf(corr_null);
    thre = interp1(yy,xx,per_thre);
end

function corr_null = corr_permutation_single(group1_map,group2_map,index_in_use,xres,yres,zres,window_size,sigma,permute_flag,smooth_flag) 
    % permute flag == 1  ==> permute only 1 
    % permute flag == 2  ==> permute both
    [group1_q_smooth] = randperm_smooth_vector(group1_map,xres,yres,zres,index_in_use,window_size,sigma,smooth_flag);
    if permute_flag == 2
        [group2_q_smooth] = randperm_smooth_vector(group2_map,xres,yres,zres,index_in_use,window_size,sigma,smooth_flag);
    else
        group2_q_smooth = group2_map;
    end
    corr_null = corr(group1_q_smooth(:),group2_q_smooth(:));
end

function [group1_q_smooth] = randperm_smooth_vector(group1_map,xres,yres,zres,index_in_use,window_size,sigma,smooth_flag) 
    qmaxM = length(index_in_use);
    index1 = randperm(qmaxM);
    group1_q = group1_map(index1);
    group1_xyz = zeros(xres*yres*zres,1);
    group1_xyz(index_in_use,1) = group1_q;
    group1_xyz = reshape(group1_xyz,[xres,yres,zres]);
    if smooth_flag == 1
        if zres==1
        group1_xyz_smooth = imgaussfilt(squeeze(group1_xyz),sigma,'FilterSize',window_size);
        else 
        group1_xyz_smooth = smooth3(group1_xyz,'gaussian',window_size,sigma);
        end
    else
        group1_xyz_smooth = group1_xyz;
    end
    group1_q_smooth = group1_xyz_smooth(index_in_use);
end

%% functions compute quantitative measurments in each group;
function [each_group_avg_dominant_caps,each_group_avg_dominant_caps_zscore,TF,consistency,ind_t,max_corr_t] = ...
    caps_group_analysis_quantative_measurement_calculation(dominant_cap_map,Ydata_t)
%==============================================================================
% cluster all points based on spatial correlation to the dominante CAPs in PD and NC each subject separately;
% each time frames only belongs to one cluster; 
% calculate Temporal frequency, switching rate, consistency for eachdCAPs in PD and NC
%=================================================================================
    [qmaxM,count_dominant_caps] = size(dominant_cap_map);
    if size(Ydata_t,1) == qmaxM
        Ydata_t = Ydata_t';
    end
    corr_Ydata_t_dominant_cap = corr(dominant_cap_map,Ydata_t');
    if count_dominant_caps == 1
        max_corr_t = corr_Ydata_t_dominant_cap;
        ind_t = ones(size(Ydata_t,1),1);
    else
        [max_corr_t,ind_t] = max(corr_Ydata_t_dominant_cap);
    end
    TF = zeros(count_dominant_caps,1);
    consistency = zeros(count_dominant_caps,1);
    for j = 1:count_dominant_caps
        TF(j,1) = sum(ind_t==j)/length(ind_t);
    end
    
    each_group_avg_dominant_caps = zeros(qmaxM,count_dominant_caps);
    each_group_avg_dominant_caps_zscore = zeros(qmaxM,count_dominant_caps);
    for j = 1:count_dominant_caps
        N_points = sum(ind_t == j);
        if N_points > 1
            tmp = Ydata_t(ind_t==j,:);
            each_group_avg_dominant_caps(:,j) = mean(tmp);
            consistency(j,1) = mean(corr(tmp',each_group_avg_dominant_caps(:,j)));
            tmp_zscore = each_group_avg_dominant_caps(:,j)./((std(tmp)')/sqrt(N_points));
            each_group_avg_dominant_caps_zscore(:,j) = tmp_zscore ./ std(tmp_zscore);
        elseif N_points == 1
            tmp = Ydata_t(ind_t==j,:);
            each_group_avg_dominant_caps(:,j) = (tmp);
            consistency(j,1) = 1;
            each_group_avg_dominant_caps_zscore(:,j) = tmp ./ std(tmp);
        end
    end
end







