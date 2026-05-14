function [X, Mv, Nv] = extract_dct_features(img, zz)
% EXTRACT_DCT_FEATURES  Sliding-window 64D DCT feature matrix.
%
%   [X, Mv, Nv] = EXTRACT_DCT_FEATURES(img, zz) slides an 8x8 window over
%   img one pixel at a time. For each block it computes the 2D DCT and
%   returns the 64-vector obtained by zig-zag scanning.
%
%   Inputs:
%     img - MxN grayscale image, doubles in [0,1] (use im2double)
%     zz  - 8x8 zig-zag map from LOAD_ZIGZAG (1..64)
%
%   Outputs:
%     X  - (Mv*Nv) x 64 matrix where Mv = M-7, Nv = N-7. Row k corresponds
%          to the block whose top-left corner is at row r, column c with
%             r = mod(k-1, Mv) + 1
%             c = floor((k-1)/Mv) + 1
%          (column-major flattening of the valid grid, matching reshape).
%     Mv, Nv - dimensions of the valid grid, returned for convenience.

    [M, N] = size(img);
    Mv = M - 7;
    Nv = N - 7;

    T = dctmtx(8);                  % 8x8 orthogonal DCT matrix
    zz_lin = zz(:);                 % column-major flatten of zig-zag map

    X = zeros(Mv*Nv, 64);
    for jj = 1:Nv
        for ii = 1:Mv
            block = img(ii:ii+7, jj:jj+7);
            D = T * block * T';                  % 8x8 DCT coefficients
            v = zeros(1, 64);
            v(zz_lin) = D(:);                    % zig-zag scan
            k = (jj-1)*Mv + ii;                  % column-major linear idx
            X(k, :) = v;
        end
    end
end
