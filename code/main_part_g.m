% MAIN_PART_G  ECE 5430 Final Project, Part (g)
%
% For each class, train 5 GMMs of C=8 components with random initialization.
% That yields 5 cheetah models x 5 grass models = 25 (FG, BG) classifier
% pairings. For each pairing, evaluate P(error) on the cheetah image at
% dimensions {1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64} and plot all 25 error
% curves on one figure to expose how sensitive EM is to initialization.
%
% To run:
%   cd matlab project folder
%   addpath('code'); main_part_g

clear; clc; close all;

this_file = mfilename('fullpath');
code_dir  = fileparts(this_file);
addpath(code_dir);
project_root = fileparts(code_dir);

DATA_DIR = fullfile(project_root, 'Project Data');
OUT_DIR  = fullfile(project_root, 'outputs');
if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

% --- Hyperparameters ------------------------------------------------------
C_COMPONENTS = 8;
N_RUNS       = 5;
DIM_LIST     = [1 2 4 8 16 24 32 40 48 56 64];
MAX_ITER     = 200;
TOL          = 1e-5;
MASTER_SEED  = 42;       % Fixed seed for reproducibility. (meme seed)
                         % Each of the 5 random initializations the
                         % project requires uses a different seed derived
                         % from MASTER_SEED (FG: 42+r, BG: 142+r for
                         % r=1..5), so the inits are genuinely random
                         % relative to each other but the SET of 5 random
                         % starts is identical across runs. This is how
                         % the report's Part (g) figures and the
                         % 4.77% numbers are reproduced exactly.
                         % Change MASTER_SEED (e.g. to a different
                         % integer) to draw a fresh family of inits.

% --- Load data ------------------------------------------------------------
fprintf('============================================================\n');
fprintf(' Part (g): 25 (FG, BG) GMM pairings, C=%d each\n', C_COMPONENTS);
fprintf('============================================================\n');

S = load(fullfile(DATA_DIR, 'TrainingSamplesDCT_8_new.mat'));
FG = S.TrainsampleDCT_FG;
BG = S.TrainsampleDCT_BG;
nFG = size(FG, 1);
nBG = size(BG, 1);
prior_FG = nFG / (nFG + nBG);
prior_BG = nBG / (nFG + nBG);

% --- Train 5 GMMs per class -----------------------------------------------
fprintf('Training %d GMMs per class...\n', N_RUNS);
gmm_FG = cell(N_RUNS, 1);
gmm_BG = cell(N_RUNS, 1);
for r = 1:N_RUNS
    fprintf('  cheetah run %d/%d ... ', r, N_RUNS);
    tic;
    [w, m, s, hist_w] = gmm_em_diag(FG, C_COMPONENTS, MAX_ITER, TOL, MASTER_SEED + r);
    gmm_FG{r} = struct('w', w, 'm', m, 's', s);
    fprintf('done in %.2fs (%d iters, ll=%.1f)\n', toc, length(hist_w), hist_w(end));

    fprintf('  grass   run %d/%d ... ', r, N_RUNS);
    tic;
    [w, m, s, hist_w] = gmm_em_diag(BG, C_COMPONENTS, MAX_ITER, TOL, MASTER_SEED + 100 + r);
    gmm_BG{r} = struct('w', w, 'm', m, 's', s);
    fprintf('done in %.2fs (%d iters, ll=%.1f)\n', toc, length(hist_w), hist_w(end));
end

% --- Extract image features (full 64D) ------------------------------------
img        = im2double(imread(fullfile(DATA_DIR, 'cheetah.bmp')));
mask_truth = im2double(imread(fullfile(DATA_DIR, 'cheetah_mask.bmp'))) > 0.5;
[M, N] = size(img);
zz = load_zigzag();
fprintf('Extracting 64D features for the cheetah image ...\n');
tic;
[X, Mv, Nv] = extract_dct_features(img, zz);
fprintf('  %d valid blocks in %.2fs\n', size(X, 1), toc);

% --- Evaluate all 25 pairings at all dimensions ---------------------------
nD = length(DIM_LIST);
errors = zeros(N_RUNS, N_RUNS, nD);   % (FG run, BG run, dim index)

fprintf('Evaluating 25 pairings x %d dimensions ...\n', nD);
tic;
for i = 1:N_RUNS
    for j = 1:N_RUNS
        for d_idx = 1:nD
            D = DIM_LIST(d_idx);
            wFG = gmm_FG{i}.w;
            mFG = gmm_FG{i}.m(:, 1:D);
            sFG = gmm_FG{i}.s(:, 1:D);
            wBG = gmm_BG{j}.w;
            mBG = gmm_BG{j}.m(:, 1:D);
            sBG = gmm_BG{j}.s(:, 1:D);
            XD = X(:, 1:D);

            lpFG = gmm_log_pdf_diag(XD, wFG, mFG, sFG) + log(prior_FG);
            lpBG = gmm_log_pdf_diag(XD, wBG, mBG, sBG) + log(prior_BG);
            y = lpFG > lpBG;
            A = zeros(M, N);
            A(1:Mv, 1:Nv) = reshape(y, Mv, Nv);
            errors(i, j, d_idx) = sum(A(:) ~= mask_truth(:)) / numel(A);
        end
    end
end
fprintf('  finished in %.2fs\n', toc);

% --- Plot all 25 curves ---------------------------------------------------
fig = figure('Name', 'Part (g): 25 GMM pairings', 'Position', [100 80 1100 700]);
hold on;
cmap = lines(N_RUNS * N_RUNS);
ci = 1;
for i = 1:N_RUNS
    for j = 1:N_RUNS
        plot(DIM_LIST, squeeze(errors(i, j, :)), '-o', ...
             'Color', cmap(ci, :), 'MarkerSize', 4, 'LineWidth', 1.0);
        ci = ci + 1;
    end
end
xlabel('Number of dimensions');
ylabel('P(error)');
title(sprintf('Part (g): 25 (FG \\times BG) pairings, C = %d, EM with random init', C_COMPONENTS));
grid on;
xlim([0 65]);
saveas(fig, fullfile(OUT_DIR, 'partG_25curves.png'));
fprintf('Saved: %s\n', fullfile(OUT_DIR, 'partG_25curves.png'));

% --- Summary table --------------------------------------------------------
fprintf('\nSummary across the 25 pairings (showing min/max/mean/std):\n');
fprintf('   D    min      max      mean     std\n');
for d_idx = 1:nD
    D  = DIM_LIST(d_idx);
    e  = squeeze(errors(:, :, d_idx));
    fprintf('  %2d  %.4f  %.4f  %.4f  %.4f\n', D, min(e(:)), max(e(:)), mean(e(:)), std(e(:)));
end

fprintf('\nAll done.\n');
