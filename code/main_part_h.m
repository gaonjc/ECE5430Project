% MAIN_PART_H  ECE 5430 Final Project, Part (h)
%
% For each class, learn one GMM per number of components C in {1, 2, 4, 8,
% 16, 32}. For each C, evaluate P(error) on the cheetah image at the same
% dimension list as part (g). Plot one error curve per C.
%
% To run:
%   cd matlab project folder
%   addpath('code'); main_part_h

clear; clc; close all;

this_file = mfilename('fullpath');
code_dir  = fileparts(this_file);
addpath(code_dir);
project_root = fileparts(code_dir);

DATA_DIR = fullfile(project_root, 'Project Data');
OUT_DIR  = fullfile(project_root, 'outputs');
if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

% --- Hyperparameters ------------------------------------------------------
C_LIST    = [1 2 4 8 16 32];
DIM_LIST  = [1 2 4 8 16 24 32 40 48 56 64];
MAX_ITER  = 200;
TOL       = 1e-5;
SEED      = 42;     % Fixed seed for reproducibility.
                    % EM uses a random initialization (random training
                    % points as initial means), so different seeds give
                    % different local optima. Fixing the seed here makes
                    % the C-vs-D error table and the partH_varying_C
                    % figure reproduce exactly the values reported in
                    % the writeup (e.g. P_e = 4.93% at C=16, D=48).
                    % Cheetah class trains with rng(SEED); grass class
                    % with rng(SEED+1). Change SEED to draw a different
                    % single random initialization.

% --- Load data ------------------------------------------------------------
fprintf('============================================================\n');
fprintf(' Part (h): error vs dimension for varying mixture sizes\n');
fprintf('============================================================\n');

S = load(fullfile(DATA_DIR, 'TrainingSamplesDCT_8_new.mat'));
FG = S.TrainsampleDCT_FG;
BG = S.TrainsampleDCT_BG;
nFG = size(FG, 1);
nBG = size(BG, 1);
prior_FG = nFG / (nFG + nBG);
prior_BG = nBG / (nFG + nBG);

% --- Image features -------------------------------------------------------
img        = im2double(imread(fullfile(DATA_DIR, 'cheetah.bmp')));
mask_truth = im2double(imread(fullfile(DATA_DIR, 'cheetah_mask.bmp'))) > 0.5;
[M, N] = size(img);
zz = load_zigzag();
fprintf('Extracting 64D features ...\n');
[X, Mv, Nv] = extract_dct_features(img, zz);
fprintf('  %d valid blocks\n', size(X, 1));

% --- Train one GMM per class for each C, evaluate at all dims -------------
nC = length(C_LIST);
nD = length(DIM_LIST);
errors = zeros(nC, nD);

for ci = 1:nC
    C = C_LIST(ci);
    fprintf('\nTraining C=%d GMMs ...\n', C);

    tic;
    [wFG, mFG, sFG, hF] = gmm_em_diag(FG, C, MAX_ITER, TOL, SEED);
    fprintf('  cheetah: %d iters, ll=%.1f, %.2fs\n', length(hF), hF(end), toc);
    tic;
    [wBG, mBG, sBG, hB] = gmm_em_diag(BG, C, MAX_ITER, TOL, SEED + 1);
    fprintf('  grass  : %d iters, ll=%.1f, %.2fs\n', length(hB), hB(end), toc);

    for d_idx = 1:nD
        D = DIM_LIST(d_idx);
        XD = X(:, 1:D);
        lpFG = gmm_log_pdf_diag(XD, wFG, mFG(:, 1:D), sFG(:, 1:D)) + log(prior_FG);
        lpBG = gmm_log_pdf_diag(XD, wBG, mBG(:, 1:D), sBG(:, 1:D)) + log(prior_BG);
        y = lpFG > lpBG;
        A = zeros(M, N);
        A(1:Mv, 1:Nv) = reshape(y, Mv, Nv);
        errors(ci, d_idx) = sum(A(:) ~= mask_truth(:)) / numel(A);
    end
end

% --- Plot one curve per C -------------------------------------------------
fig = figure('Name', 'Part (h): error vs dim per C', 'Position', [100 80 1000 600]);
cmap = lines(nC);
hold on;
for ci = 1:nC
    plot(DIM_LIST, errors(ci, :), '-o', 'Color', cmap(ci, :), ...
         'LineWidth', 1.5, 'MarkerSize', 6, ...
         'DisplayName', sprintf('C = %d', C_LIST(ci)));
end
xlabel('Number of dimensions');
ylabel('P(error)');
title('Part (h): error vs. dimension for varying mixture sizes');
legend('Location', 'best');
grid on;
xlim([0 65]);
saveas(fig, fullfile(OUT_DIR, 'partH_varying_C.png'));
fprintf('Saved: %s\n', fullfile(OUT_DIR, 'partH_varying_C.png'));

% --- Summary table --------------------------------------------------------
fprintf('\nP(error) table (rows = C, columns = D):\n');
fprintf('   C \\ D');
for d_idx = 1:nD, fprintf('    %3d', DIM_LIST(d_idx)); end
fprintf('\n');
for ci = 1:nC
    fprintf('  %3d  ', C_LIST(ci));
    for d_idx = 1:nD, fprintf('  %.4f', errors(ci, d_idx)); end
    fprintf('\n');
end

fprintf('\nAll done.\n');
