function I = scatter_to_Gaussian(scatter_spots, sigma, pixel_FOV, increment_per_pixel,normalization)

    %% the scattering_spots should be center at (pixel_FOV/2, pixel_FOV/2), unit should be pixel
    A = scatter_spots*increment_per_pixel;

    z = zeros(pixel_FOV*increment_per_pixel, pixel_FOV*increment_per_pixel);
    %scatter(scatter_spots(:,1), scatter_spots(:,2));

    ndrow = size(A,1);
    zrow = size(z,1);
    zcol = size(z,2);
    zcolvec = 1 : zcol;


    for i = 1 : ndrow;
        x1i = A(i,1);
        y1i = A(i,2);
        for j = 1 : zrow;
            jdist2 = (j-x1i).^2;
            d = sqrt(jdist2 + (zcolvec - y1i).^2);
            deltaz = exp(-d.^2/sigma^2);
            dinrange = d <= 30;
            z(j, dinrange) = z(j,dinrange) + deltaz(dinrange);
        end
    end
    
    if normalization == 'Y'
        I = z/max(max(z));
    else
        I = z;
    end
    
    
end