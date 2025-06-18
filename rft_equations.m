D = 1;
S = 1000;
pp = [0.001, 0.01, 0.05, 0.1];
u = spm_invNcdf(1-pp, 0, 1);
W =  0.1;%sqrt(mean(diff(data).^2))/sqrt(4*log(2));
Em = S*(2*pi).^(-(D+1)/2)*W.^(-D)*u.^(D-1).*exp(-u.^2./2) % expected number of clusters

EN = (1-spm_Ncdf(u,0,1))*S % expected number of voxels

En = EN./Em % expected cluster size
%%