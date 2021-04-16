folder_location = '/Users/hl/Desktop/';
save_name = ' ';
close all
warning('off','all')


cd(folder_location);
SN_ratio = 1;
origami_spacing = 84; % 84 or 28
addpath([folder_location,'function_analysis/sub_functions/ypml110-dbscan-clustering/YPML110 DBSCAN Clustering/DBSCAN Clustering']);
addpath('/Users/hl/Documents/MATLAB/MATLAB-STORM/ypml110-dbscan-clustering/YPML110 DBSCAN Clustering/DBSCAN Clustering');

min_dis = 0.005; % minumum distance (pixel) between clusters
photon_threshold = 0; % photon threshold
peak_threshold = 1000; % photon peak threshold
start_frame = 00; % start frame 

histogram_increment = 2.5;

Gaussian_radius = 3;

limit_number = 1000; % upper limit of number of origami
skip_number = 00; % skipping the number for counting


save_histogram = 'Y';
color_histogram = 'blue';

cluster_threshold = 3; % at least number of clusters
image_size = 4; %pixel*pixel
pix_size = 154; % 154nm for CMOS camera in STORM2

% parameters for DBSCAN
epsilon=0.07*pix_size;
MinPts=5;


whole_folder = 'Y';


%% choose one of the file
%% 28nm spacing
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/28nm/control/20190410/29nm-iCy5-0402_streptavidin-coverslip_5X_2D/'];
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/28nm/PA_wo-denature/20181219/Acry-29nm_2nd-iCy5_25gel-high-salt_no-digest_5X_2D/']; %.../

%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/28nm/PA_denature/20181219/Acry-29nm_2nd-iCy5_25gel-high-salt_60formamide-2M-NaCl-digest-1hr_5X_2D/']; %.../


%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/28nm/TG_wo-EXP/20191217/29nm-1st-Az_2nd-iCy5_tetra-wo-EXP-10mM-MgCl2_denature_GA-fixation_2D/']; %.../
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/28nm/TG-wo-EXP_wo-denature/20200915/29nm-1st-Az-iCy5_tetra-wo-EXP_wo-denature_2D/']; %.../

%2.2X
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/28nm/TG_EXP/20200203/29nm-1st-Az_2nd-LNA-iCy5-blocked_tetra-RT-10Mg_denature_2D/']; %.../
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/28nm/TG_EXP/20200704/29nm-1st-Az_2nd-LNA-iCy5-blocked_tetra-EXP-10Mg_light-activate-half-APS_2D/']; %.../
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/28nm/TG_EXP/20200705/29nm-1st-Az_2nd-LNA-iCy5-blocked_tetra-EXP-10Mg_light-activate-half-APS_2D/']; %.../

%3Cy5
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/28nm/supplementary/3Cy5_no_digest/20190201/Acry-29nm_2nd-3Cy5_25gel-high-salt_no-digest_5X_2D/'];  
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/28nm/supplementary/3Cy5_no_digest/20190201/Acry-29nm_2nd-3Cy5_25gel-high-salt_1M-NaCl-80F-digest-1hr_5X_2D/'];  

%5T
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/28nm/supplementary/5T_wo_digest/20181231/29nm-5T-Acry_3rd-iCy5_25gel-high-salt_wo-digest_5X_2D/']; %.../


%% 84nm spacing
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/84nm/control/20200728/90nm-1st-Acry-iCy5_2D/'];
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/84nm/PA_wo_denature/20200728/90nm-1st-Acry_2nd-iCy5-blocked_25gel-high-salt_2D/']; %.../

%bin size = 5
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/84nm/PA_denature/20200728/90nm-1st-Acry_2nd-iCy5-blocked_25gel-high-salt_denature_2D/']; %.../
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/84nm/PA_denature/20200806/90nm-1st-Acry_2nd-iCy5_25gel-high-salt_denature_2D/']; %.../

%epsilon=0.1, bin size = 2.5
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/84nm/tetra_wo-EXP_wo-denature/20200923/84nm-1st-Az_2nd-LNA-iCy5_tetra-wo-EXP-wo-denature_2D/'];
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/84nm/tetra_wo-EXP/20200923/84nm-1st-Az_2nd-LNA-iCy5_tetra-wo-EXP-denature_2D/'];

%2.1X
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/84nm/tetra_EXP_2.1X/20200917/84nm-1st-Az_2nd-LNA-iCy5_tetra-EXP_2D/'];
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/84nm/tetra_EXP_2.1X/20200921/84nm-1st-Az_2nd-LNA-iCy5_tetra-EXP-wo-remove_2D/'];
%alist_folder = [folder_location, 'TG_EXM_STORM/Data_origami/84nm/tetra_EXP_2.1X/20201001/84nm-1st-Az_2nd-LNA-iCy5_tetra-EXP_2D/'];



expanded = 'Y'; % N OR Y
expansion_factor = 2.1;



if expanded == 'Y'
    image_size = 8; %pixel*pixel
    epsilon=0.1*pix_size;
end


filenames = {};

if whole_folder == 'Y'
    directory = dir(alist_folder);
    for i = 3:length(directory);
        if directory(i).name(end-2:end) == 'bin'
            filenames{end+1} = strcat(alist_folder, directory(i).name);
        end
    end
else    
filenames = {
'/Users/hl/Desktop/Origami_mlist/20191106/29nm-5Cy5_GA-reduction-before-origami_14DMSO-10mM-MgCl2-ON_2D_good/29nm-5Cy5_GA-reduction-before-origami_14DMSO-10mM-MgCl2-ON_2D_0_alist.bin'...
}
end

%%

% indice in each file and the file directories

indice = [];

cluster_file = {};

for i = 1: length(filenames)
    a = filenames{i};
    slash_index = strfind(a,'/');
    cluster_file{end+1} = [a(1:slash_index(end)),'image/', a(slash_index(end)+1:end-4)];
end

for j = 1:length(cluster_file)
    directory = dir(cluster_file{j}); % need to be checked
    a = [];
    for i = 3:length(directory);
        if directory(i).name(end-2:end) == 'png' | directory(i).name(end-2:end) == 'bmp' 
            if directory(i).name(1) == 'n' 
                a = [str2num(directory(i).name(7:end-4)); a]; % noise term
            else
                a = [str2num(directory(i).name(1:end-4)); a]; % non-noise term
            end
        end
    end
    indice = [indice; {a}];
end



TALLY_single = [];
TALLY_all = [];
TALLY_localization = [];
count = 0; % number of all the origami
cluster_N = [];
Brightness = [];
STDxy = [];
localizations = [];
first_dis = [];
second_dis = [];
alignment_variance = [];
collapse_width = [];


%% run DBSCAN and pairwise analysis

for f = 1:length(filenames)
    filename = filenames{f};
    dash_index = strfind(filename,'/');
    textfile = strcat(filename(1:dash_index(end)-1), '/cluster_centroids/',filename(dash_index(end)+1:end-4),'.txt');

    fileID = fopen(textfile,'r');
    uni_position = [];
    n = 0;
    while true;
        tline = fgetl(fileID);
        A = findstr(tline,' ');
        if length(A) == 0
            break
        end
        uni_position = [uni_position; [str2num(tline(1:A-1)), str2num(tline(A+1:end))]];
        n = n+1;

    end


    fclose(fileID);


    MLIST = ReadMasterMoleculeList(filename);


    c = struct2cell(MLIST);

    bg = cell2mat(squeeze(c(10,:)')); % background
    XX = cell2mat(squeeze(c(3:6,:)')); % xc, yc, h, a
    Z = cell2mat(squeeze(c(18,:))'); % zc
    X = [XX,Z]; % xc, yc, h, a, zc
    F = single(cell2mat(squeeze(c(14:15,:))')); % frame, length


    X = [XX,Z,F,bg]; % xc, yc, h, a, zc, frame, length,bg
    X = X(F(:,1) >= start_frame,:);
    X = X(X(:,3)./X(:,8) > SN_ratio,:); %cut previous frames


    %[uni_position,X] = STORM_images(filename); % get the positions of clusters and the molecular list

    index = indice{f}; % index for the file

    list_cen = X(:,1:4);

    mask_brightness = list_cen(:,4) >= photon_threshold;
    mask_brightness = list_cen(:,3) >= peak_threshold;
    list_cen(mask_brightness == 0) = 0;

    for indd = 1:length(index)

        if skip_number > count
            count = count + 1;
            continue
        end

        if count == limit_number+skip_number;
            break
        end


        ind = index(indd);
        a1 = uni_position(ind,1:2);

        row = a1(1); % Y in tiff
        col = a1(2); % X in tiff

        TEST = [];

        mask=bsxfun(@gt,list_cen(:,1:2),[col-image_size/2, row-image_size/2])& bsxfun(@lt,list_cen(:,1:2),[col+image_size/2, row+image_size/2]);
        ROI_cluster = list_cen(all(mask,2),:);
        TEST = ROI_cluster(:,1:2);


        if isempty(TEST) == 1
            ind
            continue
        end

        Xm = TEST(:,1).*pix_size; % x position
        Ym = TEST(:,2).*pix_size; % y position
        %Zm = TEST(:,3); % z position

        positions = [Xm, Ym];


        % Run DBSCAN Clustering Algorithm

        IDX=DBSCAN([Xm, Ym],epsilon,MinPts);

        % show the index of the images having only one cluster or less
        if max(IDX) <= 1 
            'one cluster'
            ind        
            continue
        end

        % Find brightness of each localizations        
        B = ROI_cluster(:,4);
        B = B(IDX~= 0);
        Brightness = [Brightness; B];


        % Find centroids

        k=max(IDX);
        cen = [];

        for j = 1:k
            % get center positions of clusters
            A = IDX == j;
            S = sum(A);
            xc = dot(TEST(:,2),A)/S;
            yc = dot(TEST(:,1),A)/S;
            cen = [cen; [xc,yc]]; 

            % get standard deviations and localization numbers of clusters
            ax = (Xm).*A;
            ax = ax(ax~=0);
            ay = (Ym).*A;
            ay = ay(ay~=0);
            STDxy = [STDxy; [std(ax),std(ay)]];
            localizations = [localizations; sum(A)];
        end



        cen_dis = pdist2(cen,cen); % matrix for the distances between clusters

        re_cen_dis = reshape(cen_dis, [length(cen)^2,1]); % reshape the distances matrix


        % sort the distances and eliminate the 0 and repeat ones
        re_cen_sort = unique(re_cen_dis); 
        re_cen_sort = re_cen_sort(re_cen_sort ~= 0);

        % print if distance is small
        if re_cen_sort(1) * pix_size < 50
            [ind,re_cen_sort(1) * pix_size];
        end

        if length(re_cen_sort) < (cluster_threshold - 1) % get at least three clusters
            continue
        end

        if re_cen_sort(1) < min_dis % set minimum distance (pixel) for the clusters
            re_cen_sort(1)
            continue
        end

        % shortest distance of each cluster to the others
        shortest = sort(cen_dis);
        shortest = unique(shortest(2,:));
        first_dis = [first_dis, shortest]; % closest distance of clusters


        if cluster_threshold > 2
            second_dis = [second_dis; re_cen_sort(2)]; % second closest distance of clusters
        end

        tally_nm = re_cen_dis .* pix_size; % pairwise distances between the clusters



        idxM = cen(:,1) == max(cen(:,1)); % index of the cluster with maximum x
        tally = cen_dis(:,idxM) .* pix_size; % get only the distance between the cluster having maximum x with others

        TALLY_single = [TALLY_single; tally];
        TALLY_all = [TALLY_all; tally_nm];
        count = count + 1;
        cluster_N = [cluster_N;max(IDX)];



        %----------------------------------------------------------------------
        % find the projection axis by connecting the two more distant clusters
        %zero_cluster = cen(1,:);

        idxM = cen(:,2) == max(cen(:,2)); % index of the cluster with maximum y
        idxm = cen(:,2) == min(cen(:,2)); % index of the cluster iwth minimum y

        xx = dot(cen(:,1),idxM)-dot(cen(:,1),idxm); % x distance between the two 
        yy = dot(cen(:,2),idxM)-dot(cen(:,2),idxm); % y distance between the two

        axis_matrix = [xx/(xx^2+yy^2)^(1/2); yy/(xx^2+yy^2)^(1/2)]; % nomalized vector of the axis

        Xm_in_cluster = Xm;%(IDX~=0);
        Ym_in_cluster = Ym;%(IDX~=0);

        normal_axis_matrix = [axis_matrix(2); -axis_matrix(1)];
        project_along_axis = ([Xm_in_cluster-Xm_in_cluster(1),Ym_in_cluster-Ym_in_cluster(1)]*normal_axis_matrix);
        fg = figure('Visible','off');
        b = histogram(project_along_axis, 'BinWidth', 5);

        hist_value = b.Values;
        VB = b.BinEdges;
        hist_distance = VB(1:end-1) + b.BinWidth/2;

        try
            f1 = fit(hist_distance',hist_value',['gauss' int2str(1)]);

            collapse_width = [collapse_width; f1.c1];
        end

    end
end

TALLY_all = TALLY_all(TALLY_all>0);

square_STD = STDxy.^2;

%remove nan
nnan_X = ~isnan(square_STD(:,1));
nnan_Y = ~isnan(square_STD(:,2));
nn_all = nnan_X+nnan_Y == 2; % index of non-nan
square_STD = square_STD(nn_all,:);
localizations = localizations(nn_all,:);

variance = square_STD'*localizations;
total_STD = (variance./sum(localizations)).^(1/2);




%%
if save_histogram == 'Y'
    f = figure('Visible','on');
end


TALLY_all_adjusted = TALLY_all;
first_dis_adjusted = first_dis;


if isequal(expanded,'Y')
    TALLY_all_adjusted = TALLY_all/expansion_factor;
    first_dis_adjusted = first_dis/expansion_factor;
end

h = histogram(TALLY_all_adjusted, [histogram_increment:histogram_increment:300]);


h.FaceColor = [0.3 0.3 0.3];
set(gca,'fontsize',20)
ylabel('Counts', 'FontSize', 30);
xlabel('Pairwise peak distance (nm)', 'FontSize', 30);

%title(['Number of origami = ' num2str(count) ]);

foldername = strcat('%s_%d%s.bmp');
bri = round(mean(Brightness)/4);
B = 'B';
T = sprintf(foldername, save_name,bri, B);
%saveas(f, T);


%% save properties

text_file = strcat(save_name,'.txt');
fileID = fopen(text_file,'w');
fprintf(fileID, 'indice ={\n');
count_text = 0
for i = 1:length(indice)
    fprintf(fileID,'[');
    element = cell2mat(indice(i));
    for k = 1:length(element)             
        if count_text == limit_number;
            break
        end
        count_text = count_text +1;
        fprintf(fileID,'%d,',element(k));

    end
    fprintf(fileID,'],...\n');
end
fprintf(fileID, '}\n\n\n');

fprintf(fileID, 'filenames = {\n');
for j = 1:length(filenames)
    fprintf(fileID, cell2mat(filenames(j)));
    fprintf(fileID,',...\n');
end
fprintf(fileID, '}\n\n\n');

fprintf(fileID, 'epsilon = %.3f \n',epsilon/pix_size);
fprintf(fileID, 'MinPts = %d \n',MinPts);
fprintf(fileID, 'photon_threshold = %d \n',photon_threshold); % photon threshold

fclose(fileID);



%% Number of clusters on an origami

f = figure('Visible','on');
hold on;

h_N = histogram(cluster_N);
%hh.BinEdges = hh.BinEdges - hh.BinWidth/2;

%set(plot_f(1),'LineWidth',2);
%plot(ffit,x,y,'o');


xlabel('Clusters on an origami', 'FontSize', 20);
ylabel('Counts', 'FontSize', 20);
title(['histogram of clusters'])

foldername = strcat('%s.bmp');
bri = round(mean(Brightness)/4);
B = 'fit';
T = sprintf(foldername, save_name);
%saveas(f, T);



%% Gaussian fitting of first distance


%histogram_increment = 5;

f = figure('Visible','on');
hold on;
h_first = histogram(first_dis_adjusted*pix_size,[histogram_increment:histogram_increment:300]);


%17/45
%23 for 5nm

if histogram_increment == 5
    hist_boundary = 23;
elseif origami_spacing == 28
    hist_boundary = 17;
else
    hist_boundary = 45;
end
x = h_first.BinEdges(1:length([histogram_increment:histogram_increment:histogram_increment*hist_boundary])-1) + histogram_increment/2;
y = h_first.Values(1:length(x));
ffits = fit(x.',y.','gauss1');
%hh.BinEdges = hh.BinEdges - hh.BinWidth/2;

plot_f = plot(ffits);
set(plot_f(1),'LineWidth',2);
%plot(ffit,x,y,'o');


xlabel('Nearest neighbor distance (nm)', 'FontSize', 20);
ylabel('Counts', 'FontSize', 20);
%title(['histogram of shortest distances, number of origami = ' num2str(count) ]);

xlim([0 (floor(max(x)/100)+1)*100]);

%xticks([0 50 100 150 200]);

%xlim([0 100]);
%xticks([0 25 50 75 100]);

%ylim([0 20]);
%yticks([0 30 60]);
figure_format(h_first);


foldername = strcat('%s_%.2f_%.2f.pdf');
B = 'fit';
b1 = round(ffits.b1,2);
c1 = round(ffits.c1,2);
T = sprintf(foldername, save_name, b1, c1);
exportgraphics(f,T,'ContentType','vector');






