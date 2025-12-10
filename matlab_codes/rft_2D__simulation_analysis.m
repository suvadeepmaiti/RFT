%% rft_simulation_analysis_2D.m
% 2D RFT Monte Carlo: empirical vs theoretical cluster counts & sizes

spm('defaults','FMRI');
rng(42);  

% PARAMETERS
nx           = 64;                     % x dimension (voxels)
ny           = 64;                     % y dimension (voxels)
n_iter_W     = 500;                    % sims to estimate smoothness
n_iter       = 1000;                   % sims per FWHM & threshold
fwhm_levels  = [1, 3, 5];              % 2D smoothness levels (in voxels)
pvals        = [0.001, 0.01, 0.05, 0.1]; % cluster-forming p-values
u            = spm_invNcdf(1 - pvals, 0, 1); % corresponding Z thresholds
A            = nx * ny;                % "area" in voxels
D            = 2;                      % dimensionality
output_dir   = 'results_2D';
if ~exist(output_dir,'dir'), mkdir(output_dir); end


summary = {};

%% Helper: build 1D Gaussian kernel for separable smoothing
gauss_kernel_1D = @(FWHM) ...
    ( ...
        (FWHM == 0) * 1 + ...  % if FWHM = 0, no smoothing
        (FWHM > 0)  * ...
        ( ...
            ( ...
                exp(-(-ceil(4*(FWHM/2.3548)):ceil(4*(FWHM/2.3548))).^2 ...
                    /(2*(FWHM/2.3548)^2)) ...
            ) ...
        ) ...
    );

%% MAIN LOOP: For each smoothness level
for fi = 1:numel(fwhm_levels)
    FWHM = fwhm_levels(fi);
    fprintf('Estimating smoothness for FWHM = %g (2D)...\n', FWHM);

    % --- 1) Empirical smoothness estimates Wx, Wy ---
    Wx_samples = zeros(n_iter_W, 1);
    Wy_samples = zeros(n_iter_W, 1);

    for i = 1:n_iter_W
        X = randn(nx, ny);  % white noise field

        if FWHM > 0
            sigma  = FWHM / 2.3548;
            t      = ceil(4*sigma);
            xg     = -t:t;
            k1     = exp(-xg.^2/(2*sigma^2));
            k1     = k1 / sum(k1);
            % separable 2D convolution
            X = conv2(conv2(X, k1, 'same'), k1', 'same');
        end

        % standardise to mean 0, variance 1
        X = X - mean(X(:));
        X = X / std(X(:));

        % finite difference estimates along x and y
        dXdx = diff(X, 1, 1);   % (nx-1) x ny
        dXdy = diff(X, 1, 2);   % nx x (ny-1)

        Wx_samples(i) = sqrt(mean(dXdx(:).^2));
        Wy_samples(i) = sqrt(mean(dXdy(:).^2));
    end

    Wx_est = mean(Wx_samples);
    Wy_est = mean(Wy_samples);

    fprintf('  Estimated Wx = %.4f, Wy = %.4f\n', Wx_est, Wy_est);

    % --- 2) For each threshold, empirical & theoretical calculations ---
    for pj = 1:numel(pvals)
        uj = u(pj);
        fprintf('  Threshold Z = %.2f\n', uj);

        % Empirical cluster counts & sizes
        counts            = zeros(n_iter,1);  % #clusters per field
        vox_per_field     = zeros(n_iter,1);  % suprathreshold voxels per field
        all_cluster_sizes = [];               % pooled over all fields

        for k = 1:n_iter
            X = randn(nx, ny);

            if FWHM > 0
                sigma  = FWHM / 2.3548;
                t      = ceil(4*sigma);
                xg     = -t:t;
                k1     = exp(-xg.^2/(2*sigma^2));
                k1     = k1 / sum(k1);
                X      = conv2(conv2(X, k1, 'same'), k1', 'same');
            end

            % standardise
            X = X - mean(X(:));
            X = X / std(X(:));

            % excursion set
            mask = X > uj;

            % 2D connected components (8-connectivity)
            cc             = bwconncomp(mask, 8);
            counts(k)      = cc.NumObjects;
            vox_per_field(k) = sum(mask(:));

            if cc.NumObjects > 0
                sizes_k = cellfun(@numel, cc.PixelIdxList);
                all_cluster_sizes = [all_cluster_sizes; sizes_k(:)];
            end
        end

        EmpMeanCount     = mean(counts);              % mean #clusters
        EmpMeanVox       = mean(vox_per_field);       % mean suprathreshold voxels
        EmpMeanClustSize = mean(all_cluster_sizes);   % mean cluster size

        % --- 3) Theoretical RFT predictions (high-threshold approximation) ---
        % Expected #voxels:
        EN_vox = A * (1 - spm_Ncdf(uj, 0, 1));  % E[N]

        % Expected #clusters ~ expected Euler characteristic dominated by top term:

        Em = A * (Wx_est * Wy_est) * (2*pi)^(-3/2) * uj * exp(-uj^2/2);


        En_size = EN_vox / Em;


        summary(end+1,:) = { ...
            FWHM, uj, Wx_est, Wy_est, ...
            EN_vox, EmpMeanVox, EmpMeanCount, ...
            Em, En_size, EmpMeanClustSize ...
        }; %#ok<SAGROW>
    end
end


T = cell2table(summary, ...
    'VariableNames', { ...
        'FWHM','Z','Wx_est','Wy_est', ...
        'Theo_vox','Emp_vox','Emp_clusters', ...
        'Theo_clusters','Theo_cluster_size','Emp_cluster_size'});

disp(T);


% Unique FWHM levels
fwhm_vals = unique(T.FWHM)';

figure; 
tiledlayout(numel(fwhm_vals),1, 'TileSpacing','compact');
for i = 1:numel(fwhm_vals)
    thisF = fwhm_vals(i);
    mask  = (T.FWHM == thisF);

    Zvals         = T.Z(mask);
    Emp_clusters  = T.Emp_clusters(mask);
    Theo_clusters = T.Theo_clusters(mask);

    % sort by Z so lines connect in order
    [Zvals_sorted, idx] = sort(Zvals);
    Emp_sorted  = Emp_clusters(idx);
    Theo_sorted = Theo_clusters(idx);

    nexttile; hold on;
    plot(Zvals_sorted, Emp_sorted,  '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
    plot(Zvals_sorted, Theo_sorted, '--s', 'LineWidth', 1.5, 'MarkerSize', 6);

    xlabel('Z threshold');
    ylabel('# clusters');
    title(sprintf('Emp vs Theo cluster count (FWHM = %g)', thisF));
    legend({'Empirical','Theoretical'}, 'Location', 'best');
    grid on;
end


figure; 
tiledlayout(numel(fwhm_vals),1, 'TileSpacing','compact');
for i = 1:numel(fwhm_vals)
    thisF = fwhm_vals(i);
    mask  = (T.FWHM == thisF);

    Zvals          = T.Z(mask);
    Emp_clustSize  = T.Emp_cluster_size(mask);
    Theo_clustSize = T.Theo_cluster_size(mask);

    % sort by Z
    [Zvals_sorted, idx] = sort(Zvals);
    Emp_sorted  = Emp_clustSize(idx);
    Theo_sorted = Theo_clustSize(idx);

    nexttile; hold on;
    plot(Zvals_sorted, Emp_sorted,  '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
    plot(Zvals_sorted, Theo_sorted, '--s', 'LineWidth', 1.5, 'MarkerSize', 6);

    xlabel('Z threshold');
    ylabel('Mean cluster size (voxels)');
    title(sprintf('Emp vs Theo cluster size (FWHM = %g)', thisF));
    legend({'Empirical','Theoretical'}, 'Location', 'best');
    grid on;
end


