

%% Initialization
spm('defaults','FMRI');
rng(42);  % for reproducibility

% PARAMETERS
length       = 1000;                   % field length (voxels)
n_iter       = 1000;                   % number of simulations per FWHM
fwhm_levels  = [1, 10, 20];            % smoothness levels
pvals        = [0.001, 0.01, 0.05, 0.1];% cluster‐forming p-values
u            = spm_invNcdf(1 - pvals, 0, 1); % corresponding Z thresholds
S            = length;                 % field length for RFT formulas
D            = 1;                      % dimensionality
output_dir   = 'results';
if ~exist(output_dir,'dir'), mkdir(output_dir); end

% STORAGE FOR SUMMARY
summary = {};

%% MAIN LOOP: For each smoothness level
for fi = 1:numel(fwhm_levels)
    FWHM = fwhm_levels(fi);
    fprintf('Estimating smoothness for FWHM = %g...\n', FWHM);

    % 1) Empirical smoothness estimate W
    W_samples = zeros(n_iter,1);
    for i = 1:n_iter
        x = randn(length,1);
        if FWHM > 0
            sigma = FWHM / 2.3548;
            t     = ceil(4*sigma);
            kernel = exp(-(-t:t)'.^2/(2*sigma^2));
            kernel = kernel / sum(kernel);
            x = conv(x, kernel, 'same');
        end
        x = (x - mean(x)) / std(x);
        W_samples(i) = sqrt(mean(diff(x).^2));
    end
    W_est = mean(W_samples);

    % 2) For each threshold, do empirical & theoretical calculations
    for pj = 1:numel(pvals)
        uj = u(pj);
        fprintf('  Threshold Z = %.2f\n', uj);

        % Empirical cluster counts & sizes
        counts    = zeros(n_iter,1);
        all_sizes = [];
        clust_size = []
        for k = 1:n_iter
            x = randn(length,1);
            if FWHM > 0
                sigma = FWHM / 2.3548;
                t     = ceil(4*sigma);
                kernel = exp(-(-t:t)'.^2/(2*sigma^2));
                kernel = kernel / sum(kernel);
                x = conv(x, kernel, 'same');
            end
            x = (x - mean(x)) / std(x);
            mask = x > uj;
            cc   = bwconncomp(mask,4);
            counts(k) = cc.NumObjects;
            all_sizes = [all_sizes sum(x>uj)] ;
            clust_size = [clust_size, mean(cellfun(@length,cc.PixelIdxList),"omitnan")];
        end
        EmpMeanCount = mean(counts);
        EmpMeanSize  = mean(all_sizes);
        EmpClustSize = mean(clust_size,"omitnan");

        % Theoretical RFT predictions
        EN_vox   = S * (1 - spm_Ncdf(uj,0,1));                                        % E[N]
        Em       =  S*W_est * (2*pi)^(-(D+1)/2) *  uj^(D-1) * exp(-uj^2/2);      % E[m]
        En_size  = EN_vox/Em;                                                      % E[n]

        % Append to summary
        summary(end+1,:) = {FWHM, uj, W_est,EN_vox, EmpMeanSize,EmpMeanCount, Em, En_size,EmpClustSize}; %#ok<SAGROW>

%         figure('Visible','off');
%         histogram(counts, 'Normalization','count', ...
%                   'FaceColor',[0 0.4470 0.7410], 'EdgeColor','k');
%         hold on;
%         lambda = EmpMeanCount;
%         xvals  = 0:max(counts);
%         plot(xvals, poisspdf(xvals, lambda)*numel(counts), 'r--', 'LineWidth',1.5);
%         xlabel('Cluster Count');
%         ylabel('Frequency');
%         title(sprintf('Counts (FWHM=%g, Z=%.2f)', FWHM, uj));
%         legend('Empirical','Poisson fit');
%         saveas(gcf, fullfile(output_dir, sprintf('hist_count_FWHM%d_Z%.2f.png',FWHM,uj)));
%         close;

%         if ~isempty(all_sizes)
%             figure('Visible','off');
%             histogram(all_sizes, 'Normalization','pdf', ...
%                       'FaceColor',[0.8500 0.3250 0.0980], 'EdgeColor','k');
%             hold on;
%             pd = fitdist(all_sizes','Exponential');
%             xx = linspace(0, max(all_sizes), 200);
%             plot(xx, exppdf(xx, pd.mu), 'k--', 'LineWidth',1.5);
%             xlabel('Cluster Size (voxels)');
%             ylabel('Density');
%             title(sprintf('Sizes (FWHM=%g, Z=%.2f)', FWHM, uj));
%             legend('Empirical','Exponential fit');
%             saveas(gcf, fullfile(output_dir, sprintf('hist_size_FWHM%d_Z%.2f.png',FWHM,uj)));
%             close;
%         end
    end
end

%% SAVE SUMMARY
T = cell2table(summary, ...
    'VariableNames', {'FWHM','Z','W_est','Theo_vox','Emp_vox','Emp_clusters','Theo_clusters','Theo_cluster_size', 'Emp_cluster_size'});
% Display & write CSV
disp(T);
