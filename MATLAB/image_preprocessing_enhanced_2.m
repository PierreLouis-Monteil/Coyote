function detect_dark_on_flatfield_robust(imgPath, pixelSize_um)
% Robust dark-blob (crystal) detection with flat-fielding, local z-score,
% multi-scale, scale-normalized LoG, and statistical thresholding.
%
% Usage:
%   detect_dark_on_flatfield_robust('image.png', 0.5)    % pixel size in µm/px
%   detect_dark_on_flatfield_robust('image.png')         % defaults to 1 µm/px
%
% Output:
%   - Figures showing each stage
%   - Console stats
%   - CSV "crystal_measurements_robust.csv" with centroids & dimensions

tic;

if nargin < 1, imgPath = 'image_2_prakriti_cropped.png'; end
if nargin < 2, pixelSize_um = 1; end   % if scale unknown, stays in pixels

% ================= Tunable parameters (stable across images) ==============
bgSigma      = 40;          % Gaussian sigma for background (flat-field)
rimFrac      = 0.92;        % keep only central disk to avoid vignette rim
minAreaPx    = 100;         % remove tiny specks after thresholding
closeR       = 2;           % morphological closing radius
locSigma     = 6;           % local mean/std window (px) for z-score
scales       = [1.2 1.8 2.6 3.6 4.8];   % LoG radii (px) to cover blob sizes
alpha        = 1e-6;        % per-pixel false-positive rate (Gaussian tail)
useFDR       = false;       % true => Benjamini-Hochberg FDR instead of alpha
qFDR         = 0.05;        % FDR rate if useFDR=true
gatePercent  = 70;          % LoG gate percentile (60–80 typical)
% ==========================================================================

% ------------------ Load & grayscale ------------------
I0 = imread(imgPath);
if size(I0,3) > 1, I0 = rgb2gray(I0); end
I0 = im2single(I0);

% ------------------ Flat-field ------------------------
bg = imgaussfilt(I0, bgSigma);
F  = I0 - bg;
F  = F - min(F(:));
F  = F ./ max(F(:) + eps);  % [0,1]

% ------------------ ROI (avoid rim) -------------------
[h,w] = size(F);
[xg,yg] = meshgrid((1:w)-w/2, (1:h)-h/2);
roi = sqrt(xg.^2 + yg.^2) <= min(h,w)*0.5*rimFrac;

% ------------------ Local z-score ---------------------
mu   = imgaussfilt(F, locSigma);
varL = imgaussfilt(F.^2, locSigma) - mu.^2;
sigmaL = sqrt(max(varL, 1e-8));
Z  = (F - mu) ./ sigmaL;       % negative where locally dark
Zneg = max(-Z, 0);             % positive for dark

% ------------------ Multi-scale LoG (scale-normalized) -------------------
LoGmax = zeros(size(F), 'like', F);
for s = scales
    k = ceil(6*s); if mod(k,2)==0, k = k+1; end
    hLog = fspecial('log', [k k], s);
    resp = -(s^2) * imfilter(F, hLog, 'symmetric');  % dark centers => positive
    LoGmax = max(LoGmax, resp);
end
LoGmax = LoGmax ./ max(LoGmax(:) + eps);

% ------------------ Combine cues into score (0..1) -----------------------
Zd = Zneg; Zd = Zd ./ max(Zd(:) + eps);
score = 0.5*Zd + 0.5*LoGmax;
score(~roi) = 0;

% ------------------ Statistical thresholding -----------------------------
if ~useFDR
    % Per-pixel alpha on Zneg (Gaussian right tail)
    zThr = sqrt(2) * erfcinv(2*alpha);   % e.g., alpha=1e-6 -> ~4.75
    BW1  = (Zneg >= zThr) & roi;
else
    % FDR on Zneg p-values
    p = 1 - normcdf(Zneg);   % right-tail p-values
    p(~roi) = 1;
    [ps, ord] = sort(p(:), 'ascend');
    m = numel(ps);
    thrIdx = find(ps <= ((1:m)'/m)*qFDR, 1, 'last');
    BW1 = false(size(F));
    if ~isempty(thrIdx)
        pcut = ps(thrIdx);
        BW1 = p <= pcut;
    end
    BW1 = BW1 & roi;
end

% Gate using LoG strength to suppress non-blob texture
tau = prctile(LoGmax(roi), gatePercent);
BW2 = (LoGmax >= tau) & roi;

BW = BW1 & BW2;   % robust mask before cleanup

% ------------------ Cleanup ---------------------------
BW = imclose(BW, strel('disk', closeR));
BW = imfill(BW, 'holes');
BW = bwareaopen(BW, minAreaPx);

% ------------------ Measure blobs ---------------------
CC = bwconncomp(BW);
S  = regionprops(CC, 'Area','Perimeter','Centroid','BoundingBox', ...
                      'EquivDiameter','MajorAxisLength','MinorAxisLength');

T = struct2table(S);
if ~isempty(T)
    % Pixels
    T.CentroidX_px = T.Centroid(:,1);
    T.CentroidY_px = T.Centroid(:,2);
    T.Circularity  = 4*pi*T.Area ./ (T.Perimeter.^2 + eps);
    T.AxisRatio    = T.MinorAxisLength ./ (T.MajorAxisLength + eps);
    % Microns
    T.CentroidX_um       = T.CentroidX_px * pixelSize_um;
    T.CentroidY_um       = T.CentroidY_px * pixelSize_um;
    T.EquivDiameter_um   = T.EquivDiameter   * pixelSize_um;
    T.MajorAxisLength_um = T.MajorAxisLength * pixelSize_um;
    T.MinorAxisLength_um = T.MinorAxisLength * pixelSize_um;
    T.Area_um2           = T.Area * (pixelSize_um^2);
end
toc;

% ------------------ Plots ------------------------------
figure('Name','Robust dark-blob detection');
tiledlayout(3,4,'Padding','compact','TileSpacing','compact');

nexttile; imshow(I0,[]); title('1) Original');
nexttile; imshow(bg,[]);  title('2) Background');
nexttile; imshow(F,[]);   title('3) Flat-fielded');

nexttile; imshow(Zneg,[]); title('4) Local z-score (dark ↑)');
nexttile; imshow(LoGmax,[]); title('5) Max LoG (multi-scale)');
nexttile; imshow(score,[]); title('6) Combined score');

nexttile; imshow(BW1,[]); title('7) Stat. mask (alpha/FDR)');
nexttile; imshow(BW2,[]); title(sprintf('8) LoG gate ≥ %d%%',gatePercent));
nexttile; imshow(BW,[]);  title('9) Union & cleanup');

% Overlay detections
nexttile; imshow(I0,[]); hold on;
for k=1:height(T)
    rectangle('Position', T.BoundingBox(k,:), 'EdgeColor','r','LineWidth',1.5);
    p = plot(T.CentroidX_px(k), T.CentroidY_px(k), 'g+'); set(p,'MarkerSize',7,'LineWidth',1.2);
end
title(sprintf('10) Detections: %d', height(T))); hold off;

% Confidence map = score × BW
conf = score .* BW;
nexttile; imshow(conf,[]); title('11) Confidence (score×mask)');

% Size histograms
nexttile; 
if ~isempty(T)
    histogram(T.EquivDiameter); xlabel('Equiv. diameter (px)'); ylabel('Count');
else
    text(0.5,0.5,'No detections','HorizontalAlignment','center'); axis off;
end
title('12) Size distribution');

% ------------------ Stats & export ---------------------
fprintf('Robust detector: %d crystals\n', height(T));
if ~isempty(T)
    fprintf('Median equiv. diam = %.2f px (%.2f µm)\n', ...
        median(T.EquivDiameter), median(T.EquivDiameter_um));
    fprintf('Median area = %.1f px^2 (%.2f µm^2)\n', ...
        median(T.Area), median(T.Area_um2));
    writetable(T, 'crystal_measurements_robust.csv');
    disp('Saved: crystal_measurements_robust.csv');
end


end
