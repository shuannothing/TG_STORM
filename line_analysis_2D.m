folder_location = '/Users/hl/Desktop/';

addpath([folder_location,'function_analysis/sub_functions']);
addpath([folder_location,'function_analysis/sub_functions/FastPeakFind']);


close all




%% choose one of the file
scattering_path = [folder_location,'TG_EXM_STORM/Data_origami/2D origami/20200704/2D-10nm-1st-Az_2nd-LNA-iCy5_tetra-EXP-10Mg-wo-remove-coverslip_2D/analyze_peakfinding_th=0.3'];
%scattering_path = [folder_location,'TG_EXM_STORM/Data_origami/2D origami/20200705/2D-10nm-1st-Az_2nd-LNA-iCy5-blocked_tetra-EXP-10Mg_light-activate-half-APS_2D/analyze_peakfinding_th=0.3'];
%scattering_path = [folder_location,'TG_EXM_STORM/Data_origami/2D origami/20200812/combine_2D-10nm-1st-Az_2nd-iCy5-LNA-block-wash_tetra-wo-remove_half-APS_2D/analyze_peakfinding_th=0.3'];

%%
Gaussian_radius = 3;

rendering_radius = 8;

pix_size = 153;
expanded = 'Y';


Blur_radius = 3;

distance_max = 20; % cleaning signals outside of this radius from the clusters for K-means %15/20

find_peak_threshold = 0.3;

% screening the first peak
screen_first_peak = 'N';
first_peak_low = 45;
first_peak_high = 58;


line_distance_low = 65;
line_distance_high = 75;

histogram_increment = 1.5;


cluster_threshold = 3; % at least number of clusters (3)
image_size = 1; %pixel*pixelx


EXP_factor = 1;


if expanded == 'Y';
    image_size = 2; %pixel*pixelx %2
    EXP_factor = 2.5; %2.5
    
end



if expanded == 'Y'
    line_distance = 80; %30, 80
    distance_interval = 5; %2.5, 5
    distance_lower_bound = 50; %20, 50
    distance_upper_bound = 100; % 40, 100
    bin_size = 2.5; %1, 2.5

else
    line_distance = 30; %30, 80
    distance_interval = 2.5; %2.5, 5
    distance_lower_bound = 20; %20, 50
    distance_upper_bound = 40; % 40, 100
    bin_size = 1; %1, 2.5
end


%%


cd(scattering_path);
% find selected images
cluster_file = scattering_path;

directory = dir(cluster_file); % need to be checked
image_names = {};
for i = 3:length(directory);
    if directory(i).name(end-2:end) == 'png' | directory(i).name(end-2:end) == 'bmp' 
        if directory(i).name(1) == 'n' 
            image_names{end+1} = directory(i).name; % noise term
        else
            image_names{end+1} = directory(i).name; % non-noise term
        end
    end
end
% finding the matching name


TALLY_all = [];
localization_pairwise_all = [];
count = 0; % number of all the origami
cluster_N = [];
Brightness = [];
STDxy = [];
localizations = [];
first_dis = [];
second_dis = [];
alignment_variance = [];
collapse_width = [];
line_distance_hist = [];
cluster_N = [];


I0 = scatter_to_Gaussian([], Gaussian_radius, image_size, pix_size,'N');

index_ = findstr(scattering_path,'/');
cd(scattering_path(1:index_(end)-1));

%%
for image_ind = 1:length(image_names)
    image_name = image_names{image_ind};


    text = strcat(image_name(1:end-3),'txt');
    fileID = fopen(text,'r');
    n = 0;
    scattering = [];
    while true;
        tline = fgetl(fileID);
        A = findstr(tline,' ');

        if length(A) == 0
            break
        end

        scattering = [scattering; [str2num(tline(1:A-1)), str2num(tline(A+1:end))]];
        n = n+1;
    end
    fclose(fileID);


    % render image

    try
        [image_2D, residue, optimized_line_distance, scattering_11, scattering_22] = two_line_fitting('', scattering, '', line_distance, 11, 0.1, 5, 1, distance_interval,image_size,'None1',rendering_radius);
    catch 
        continue
    end
    

        
    % RUN Kmeans

    %%

    image_2D = image_2D/max(max(image_2D));
    image_2D = image_2D-find_peak_threshold;

    peak_in = FastPeakFind(image_2D);
    peak_ind = [];

    for i = 1:length(peak_in)/2
        peak_ind = [peak_ind;peak_in(2*i-1), peak_in(2*i)];
    end

    image_2D_normalized = image_2D/max(max(image_2D))/2;

    for j = 1:length(peak_in)/2
        image_2D_normalized(peak_ind(j,2),peak_ind(j,1)) = 1;
    end

    peak_1 = peak_ind(peak_ind(:,1)<image_size*pix_size/2,:);
    peak_2 = peak_ind(peak_ind(:,1)>image_size*pix_size/2,:);



    for j = 1:length(peak_in)/2
        image_2D_normalized(peak_ind(j,2),peak_ind(j,1)) = 1;
    end

    % K-means with initial guess based on the peak-finding
    scattering_combine = [scattering_11;scattering_22];

    peak_ind_intial = [peak_ind(:,1)-image_size*pix_size/2,image_size*pix_size/2-peak_ind(:,2)];
    distance_to_peaks = pdist2(scattering_combine,peak_ind_intial);
    distance_to_peaks = sum(distance_to_peaks<distance_max,2);
    scattering_combine = scattering_combine(distance_to_peaks>0,:);
    num_cluster = size(peak_ind,1);
    [idx,C] = kmeans(scattering_combine,num_cluster,'Start', peak_ind_intial);


    same_FOV_scattering = {C(C(:,1)<0,:),C(C(:,1)>0,:)};



    line_in_origami = 0;
    for same_FOV_scattering_ind = 1:2
        TEST = same_FOV_scattering{same_FOV_scattering_ind}/pix_size;

        FOV_scattering = {scattering_11,scattering_22};
        scattering_line = FOV_scattering{same_FOV_scattering_ind}/pix_size;
        scattering_line = scattering_line-mean(TEST);
        TEST = TEST-mean(TEST);
        
        % screen the line with few clusters
        if size(TEST,1) < cluster_threshold
            continue
        else        
            cluster_N = [cluster_N;size(TEST,1)];
            cen = TEST;
        end

        theta = pi/2;

        cen = cen * [cos(theta) sin(theta); sin(theta) -cos(theta)];
        TEST = TEST * [cos(theta) sin(theta); sin(theta) -cos(theta)];

        % make the first cluster align at pixel/4 at the left
        [sort_cen,sort_cen_I] = sort(cen(:,1));
        
        
        TEST(:,1) = TEST(:,1) - sort_cen(2); 
           
        
        scattering_line = scattering_line*[cos(theta) sin(theta); sin(theta) -cos(theta)];
        
        
        fg = figure('Visible','off');
        b = histogram(scattering_line(:,2)*pix_size, 'BinWidth', 1);
        hist_value = b.Values;
        VB = b.BinEdges;
        hist_distance = VB(1:end-1) + b.BinWidth/2;

        try
            f1 = fit(hist_distance',hist_value','gauss1');

            collapse_width = [collapse_width; f1.c1];

        end
        
        
  
        



        if isequal(screen_first_peak,'Y')
            if sum(pairwise_locs > (first_peak_low) & pairwise_locs < (first_peak_high)) < 1 % screen the peak between selected first_peak range                
                continue
            end
        end
        cen_dis = pdist2(cen,cen); % matrix for the distances between clusters


        shortest = sort(cen_dis);
        shortest = unique(shortest(2,:));
        first_dis = [first_dis, shortest]; % closest distance of clusters



       
        count = count + 1;
        line_in_origami = line_in_origami +1;
    end
    if line_in_origami > 0
        line_distance_hist = [line_distance_hist; optimized_line_distance];
    end

end



%%


first_dis_adjusted = first_dis;

if isequal(expanded,'Y')
    TALLY_all_exp = TALLY_all/EXP_factor;
    first_dis_adjusted = first_dis/EXP_factor;
end

f_hist = figure
h = histogram(TALLY_all_exp, [histogram_increment:histogram_increment:60]);

hh_1nm = histcounts(TALLY_all_exp, [1:1:60]);



hh = histogram(TALLY_all_exp, [histogram_increment:histogram_increment:60]);

hist_1_Gaussian = imgaussfilt(hh_1nm, 1.5);

hold on

blur_plot = plot(1.5:1:59.5,hist_1_Gaussian);
set(blur_plot,'LineWidth',2);

set(gca,'fontsize',20)
ylabel('Counts', 'FontSize', 30);
xlabel('Pairwise peak distance (nm)', 'FontSize', 30);

xlim([0 40])
xticks([0 10 20 30 40]);

figure_format(hh)


%%
localization_pairwise_all = localization_pairwise_all(localization_pairwise_all>0);
localization_pairwise_all_EXP = localization_pairwise_all;
if isequal(expanded,'Y')
    localization_pairwise_all_EXP = localization_pairwise_all/EXP_factor;
end

figure
h = histogram(localization_pairwise_all_EXP, [histogram_increment:histogram_increment:60]);



set(gca,'fontsize',20)
ylabel('Counts', 'FontSize', 30);
xlabel('Localization pairwise distance (nm)', 'FontSize', 30);


foldername = strcat('%s_%s.pdf');


%% line distance
figure
dis_hist = histogram(line_distance_hist/EXP_factor, [18.75:2.5:41.25]);

set(gca,'fontsize',20)
ylabel('Counts', 'FontSize', 30);
xlabel('Line distance (nm)', 'FontSize', 30);
figure_format(dis_hist);




%%
figure
h_width = histogram(collapse_width/EXP_factor, [0:2:20]);

xlim([0 20])
ylim([0 max(h_width.Values)*1.2]);
set(gca,'fontsize',20)
xlabel('Individual line widths (nm)', 'FontSize', 30);
ylabel('Count', 'FontSize', 30);

figure_format(h_width);

mean(collapse_width(collapse_width/EXP_factor<30)/EXP_factor)

%% Gaussian fitting of first distance
        

f = figure('Visible','on');
hold on;
h_first = histogram(first_dis_adjusted*pix_size,[histogram_increment:histogram_increment:40]);

% 17 12
x = h_first.BinEdges(1:length([histogram_increment:histogram_increment:histogram_increment*12])-1) + histogram_increment/2;
y = h_first.Values(1:11);
ffits = fit(x.',y.','gauss1');

plot_f = plot(ffits);
set(plot_f(1),'LineWidth',2);

xlabel('Nearest neighbor distance (nm)', 'FontSize', 20);
ylabel('Counts', 'FontSize', 20);
%title(['histogram of shortest distances, number of origami = ' num2str(count) ]);
xlim([0 40])
ax = gca;
xticks([0:10:40])
figure_format(h_first);

foldername = strcat('%s_%.2f_%.2f.pdf');
b1 = round(ffits.b1,2);
c1 = round(ffits.c1,2);
T = sprintf(foldername, '/Users/hl/Desktop/', b1, c1);
exportgraphics(f,T,'ContentType','vector');
