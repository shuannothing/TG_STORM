function [xy,residue, optimized_line_distance, scattering_1, scattering_2] = two_line_fitting(saving_folder, scattering, text_name, line_distance, residue_threshold, radius, MinPts, DBSCAN_threshold, distance_interval,image_size,color,rendering_radius)
 
    %% input:
    %scattering : scattering data centered at zero (unit: nm)
    
    %rendering_radius = 8;
    %scattering = normal_TEST;
    %line_distance = 80;
    %radius = 0.1;

    pix_size = 153;
    epsilon = radius * pix_size;
    %MinPts = 5;
    %DBSCAN_threshold = 1;
    distance_interval = 2.5;
    %saving_folder = '';
    %text_name = '';
    %residue_threshold = 11;
    %color = 'None1';

    addpath('/Users/hl/Documents/MATLAB/MATLAB-STORM/ypml110-dbscan-clustering/YPML110 DBSCAN Clustering/DBSCAN Clustering');
    %image_size = 2; %pixel*pixel

    
    IDX=DBSCAN(scattering,epsilon,MinPts);

    if max(IDX) < DBSCAN_threshold
        f2 = 0; 
        T = 0; 
        residue = 0;
        optimized_line_distance = 0;
        return 
    end
    
    

    % fitting two lines within the image
    m_ix = [0,0]; 
    minimum_distance = 1000000;
    for i = 0:2.5:180 % different angle
        for x_move = -200:2.5:200 % movement of the lines
            a = abs(scattering(:,2)-tand(i).*scattering(:,1)-(x_move+line_distance/2)/cosd(i))/(1+(tand(i))^2)^(1/2);
            b = abs(scattering(:,2)-tand(i).*scattering(:,1)-(x_move-line_distance/2)/cosd(i))/(1+(tand(i))^2)^(1/2);
            sum_min = sum(min([a,b],[],2));

            if minimum_distance > sum_min
                minimum_distance = sum_min;          
                m_ix = [i, x_move];
            end

        end
    end
    
    optimized_line_distance_increment = 0;
    final_minimum_distance = minimum_distance;
    
    %% changing line distance
    for line_distance_increment = -distance_interval*6:distance_interval:distance_interval*6
            a = abs(scattering(:,2)-tand(m_ix(1)).*scattering(:,1)-(m_ix(2)+(line_distance+line_distance_increment)/2)/cosd(m_ix(1)))/(1+(tand(m_ix(1)))^2)^(1/2);
            b = abs(scattering(:,2)-tand(m_ix(1)).*scattering(:,1)-(m_ix(2)-(line_distance+line_distance_increment)/2)/cosd(m_ix(1)))/(1+(tand(m_ix(1)))^2)^(1/2);
            sum_min = sum(min([a,b],[],2));
            
            if final_minimum_distance > sum_min
                final_minimum_distance = sum_min;    
                optimized_line_distance_increment = line_distance_increment;
            end
    end
  

    idx = ((scattering(:,2)-tand(m_ix(1)).*scattering(:,1)-m_ix(2)/cosd(m_ix(1)))>0)+1;




    
    while false
        figure('Visible','off');
        hold on
        xlim([-pix_size*image_size/2 pix_size*image_size/2]);
        ylim([-pix_size*image_size/2 pix_size*image_size/2]);
        scatter(scattering(:,1), scattering(:,2));

        axis equal;
    end
    
    
    %%
    
    angle = m_ix(1)+90;
    scattering_rotate = scattering*[cosd(angle) -sind(angle); sind(angle) cosd(angle)]; 
    residue = round(final_minimum_distance/length(scattering(:,2)),2);
    optimized_line_distance = line_distance + optimized_line_distance_increment;
    

    if residue > residue_threshold
        f2 = 0; 
        T = 0; 
        optimized_line_distance = 0;
        return
    end
    
    scattering_1 = scattering_rotate(idx==1,:);
    scattering_2 = scattering_rotate(idx==2,:);
    

    %% scattering plot of rotated image
    %f2 = figure('Visible','off');
    
    f2 = figure('Visible','off');
    
    if color == 'None1'
        close(f2)
    end
    
    if color == 'color'
        
        subplot(2,1,1)
        hold on
        xlim([-pix_size*image_size/2 pix_size*image_size/2]);
        ylim([-pix_size*image_size/2 pix_size*image_size/2]);

        line_incre = -150:150;
        %plot(line_incre,tand(m_ix(1)).*line_incre+(m_ix(2)-(line_distance + optimized_line_distance_increment)/2)/cosd(m_ix(1)));
        %plot(line_incre,tand(m_ix(1)).*line_incre+(m_ix(2)+(line_distance + optimized_line_distance_increment)/2)/cosd(m_ix(1)));

        %plot(-(line_distance + optimized_line_distance_increment)*ones(length(line_incre))/2, line_incre);
        %plot(+(line_distance + optimized_line_distance_increment)*ones(length(line_incre))/2, line_incre);
        axis equal;




        scatter(scattering_1(:,1), scattering_1(:,2));
        scatter(scattering_2(:,1), scattering_2(:,2));

        %scattering_1 = scattering_1-mean(scattering_1);
        %scattering_2 = scattering_2-mean(scattering_2);

        if length(scattering_1) > 3* length(scattering_2) ||  length(scattering_2) > 3* length(scattering_1)
            f2 = 0; 
            T = 0; 
            optimized_line_distance = 0;
            return
        end

        
        title(['Fitting residue = ', num2str(residue), ', line distance = ', num2str(optimized_line_distance), 'nm' ]);
        %axis([col-image_size/2 col+image_size/2 row-image_size/2 row+image_size/2]);
        set(gcf,'PaperUnits','inches','PaperPosition',[0 0 8 6]);
        %colormap(winter);
        colorbar;
        hold off


        subplot(2,1,2)
    end




    %% rendering images ------------------------------------

    angle = m_ix(1);
    scattering_rotate = scattering*[cosd(angle) -sind(angle); sind(angle) cosd(angle)];    

    

    xy = scatter_to_Gaussian_2D([(scattering_rotate(:,1)-mean(scattering_rotate(:,1)))/pix_size+ image_size/2,(scattering_rotate(:,2)-mean(scattering_rotate(:,2)))/pix_size+ image_size/2], rendering_radius, image_size, pix_size, color);
    xyz = xy/max(max(xy));

    imshow(xyz);
    axis equal;
    hold on
    title(['Rendering radius = ', num2str(rendering_radius)]);
    %axis([col-image_size/2 col+image_size/2 row-image_size/2 row+image_size/2]);
    set(gcf,'PaperUnits','inches','PaperPosition',[0 0 8 6]);


    hold off
   
    if color == 'None1';
        f2 = 0; 
        T = 0; 
        %optimized_line_distance = 0;
        return
    end
    
   
 
    

    foldername = strcat(saving_folder, text_name,'.png');

    T = sprintf(foldername);
    
    saveas(f2, T);
    clear T;
    cla;
    close(f2);

    
    
    
    while false

        coeff1 = pca(scattering_1*pix_size);
        angle1 = -atand(coeff1(1,2)/coeff1(1,1));
        scattering_1 = scattering_1*[cosd(angle1) -sind(angle1); sind(angle1) cosd(angle1)];

        coeff2 = pca(scattering_2*pix_size);    
        angle2 = -atand(coeff2(1,2)/coeff2(1,1));
        scattering_2 = scattering_2*[cosd(angle2) -sind(angle2); sind(angle2) cosd(angle2)]



        %% rendering clustered images ------------------------------------

        figure('Visible','on');
        axis equal;
        hold on
        subplot(2,1,1)
        xy = scatter_to_Gaussian_2D([(scattering_1(:,1)-mean(scattering_1(:,1)))/pix_size+ image_size/2,(scattering_1(:,2)-mean(scattering_1(:,2)))/pix_size+ image_size/2], rendering_radius, image_size, pix_size, color);

        title(['Rendering radius = ', num2str(rendering_radius)]);
        %axis([col-image_size/2 col+image_size/2 row-image_size/2 row+image_size/2]);
        set(gcf,'PaperUnits','inches','PaperPosition',[0 0 8 6]);

        subplot(2,1,2)
        plot(sum(xy,2)) %projection plot

        hold off

        figure('Visible','on');
        hold on
        subplot(2,1,1)
        xy = scatter_to_Gaussian_2D([(scattering_2(:,1)-mean(scattering_2(:,1)))/pix_size+ image_size/2,(scattering_2(:,2)-mean(scattering_2(:,2)))/pix_size+ image_size/2], rendering_radius, image_size, pix_size, color);
        
        
        title(['Rendering radius = ', num2str(rendering_radius)]);
        %axis([col-image_size/2 col+image_size/2 row-image_size/2 row+image_size/2]);
        set(gcf,'PaperUnits','inches','PaperPosition',[0 0 8 6]);
        colormap(jet);
        colorbar;

        subplot(2,1,2)
        plot(sum(xy,2)) %projection plot

        hold off
        
        break
    end
    
    
end