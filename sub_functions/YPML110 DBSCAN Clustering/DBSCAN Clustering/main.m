%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML110
% Project Title: Implementation of DBSCAN Clustering in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

%clc;
%clear;
%close all;
MLIST = ReadMasterMoleculeList(filename);

% Load Data


%data=load('mydata');
%X=data.X;


c = struct2cell(MLIST);

X = cell2mat(squeeze(c(3:6,:)'));

list_sig = [];

for row = 0:4
    for col = 0:4
        TEST = [];
        
        for i = 1:length(MLIST)
            if X(i,2) >= (row*120+1) && X(i,2) <= (row*120+120)
                if X(i,1)>= (col*120+1) && X(i,1) <= (col*120+120)
                    if X(i,4) >= 10 % set threshold for the brightness
                        TEST = [TEST;X(i,1:2)];
                    end
                end
            end
        end
        TEST_1 = mat2cell(TEST,[size(TEST,1)]);
        list_sig = [list_sig; TEST_1];
        
    end
end


tally_all = [];
%% Run DBSCAN Clustering Algorithm

cen_position = [];

for i = 1:9
    frag_sig = cell2mat(list_sig(i));
    epsilon=0.5;
    MinPts=5;
    IDX=DBSCAN(frag_sig,epsilon,MinPts);


    % Plot Results

    %PlotClusterinResult(TEST, IDX);
    %title(['DBSCAN Clustering (\epsilon = ' num2str(epsilon) ', MinPts = ' num2str(MinPts) ')']);


    % Find centroids

    k=max(IDX);
    cen = [];

    for j = 1:k
        A = IDX == j;
        S = sum(A);
        xc = dot(frag_sig(:,2),A)/S;
        yc = dot(frag_sig(:,1),A)/S;
        cen = [cen; [xc,yc]]; 
    end

    cen_dis = pdist2(cen,cen);

    % find the index of dyes distances < 2 pixels
    cen_origami_ind = cen_dis < 2; 
    ind = find(cen_origami_ind);
    for z = 1:length(ind)
        a = mod(ind(z), length(cen));    % remainder
        b = (ind(z) - a)/length(cen);    % dividend
        if a == 0
            cen_position = [cen_position; cen(b,1:2)];
        else
            cen_position = [cen_position; cen(b+1,1:2)];
        end
    end

    re_cen_dis = reshape(cen_dis, [length(cen)^2,1]);
    
    tally_all = [tally_all; re_cen_dis];
end

histogram(tally_all, [0.1:0.1:3]);