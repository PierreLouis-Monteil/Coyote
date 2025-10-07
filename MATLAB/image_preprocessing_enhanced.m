function detect_dark_on_flatfield_plus(imgPath, pixelSize_um)
% Flat-field + robust dark-blob detection with local contrast normalization,
% multi-scale LoG, black-hat, and adaptive threshold fallback.
% Usage: detect_dark_on_flatfield_plus('image.png', 0.45)

if nargin < 1, imgPath = 'image_2_prakriti_cropped.png'; end
if nargin < 2, pixelSize_um = 1; end

tic;

% ---- Tunables (increase sensitivity if you miss faint blobs) ----
bgSigma       = 40;        % vignette/background sigma
rimFrac       = 0.92;      % ignore vignette rim
minAreaPx     = 100;       % remove tiny specks
closeR        = 2;         % morphological closing radius
darkPercent   = 10;        % keep top X% of "darkness score" (lower => stricter)
locNormSigma  = 6;         % local mean/std window for z-score (increase if noise)
bhRadius      = 6;         % black-hat disk radius for dark blob emphasis
logSigmas     = [1.2 1.8 2.6 3.6];   % multi-scale LoG (covers a range of blob sizes)
scoreWeights  = [0.4 0.3 0.3];       % [zscore, blackhat, LoG] weights; sum≈1
adaptSens     = 0.60;      % adaptive threshold sensitivity (0..1), higher=more foreground
% -----------------------------------------------------------------

% Read & grayscale
I0 = imread(imgPath);
if size(I0,3)>1, I0 = rgb2gray(I0); end
I0 = im2single(I0);

% --- Flat-field ---
bg = imgaussfilt(I0, bgSigma);
F  = I0 - bg;
F  = F - min(F(:)); F = F ./ max(F(:)+eps);

% Optional rim mask
[h,w] = size(F);
[xg,yg] = meshgrid((1:w)-w/2, (1:h)-h/2);
roi = sqrt(xg.^2 + yg.^2) <= min(h,w)*0.5*rimFrac;

% --- Local contrast normalization (z-score) ---
mu  = imgaussfilt(F, locNormSigma);
mu2 = imgaussfilt(F.^2, locNormSigma);
sigma = sqrt(max(mu2 - mu.^2, 1e-8));
Z  = (F - mu) ./ sigma;             % darker-than-local => more negative
Zd = -Z;                            % higher = darker (locally)
Zd = Zd - min(Zd(:)); Zd = Zd ./ max(Zd(:)+eps);

% --- Black-hat (emphasize dark blobs) ---
se = strel('disk', bhRadius);
BH = imbothat(F, se);
BH = BH ./ max(BH(:)+eps);

% --- Multi-scale LoG (take max response) ---
LOGmax = zeros(size(F),'like',F);
for s = logSigmas
    k = ceil(6*s); if mod(k,2)==0, k = k+1; end
    resp = -imfilter(F, fspecial('log', [k k], s), 'symmetric'); % dark centers => positive
    LOGmax = max(LOGmax, resp);
end
LOGmax = LOGmax ./ max(LOGmax(:)+eps);

% --- Combined "darkness" score ---
wz = scoreWeights(1); wbh = scoreWeights(2); wlg = scoreWeights(3);
score = wz*Zd + wbh*BH + wlg*LOGmax;

% --- Threshold: percentile on score + adaptive local fallback ---
thScore = prctile(score(roi), 100 - darkPercent);
BW1 = score >= thScore;
BW2 = imbinarize(F, adaptthresh(F, adaptSens, 'ForegroundPolarity','dark'));
BW  = (BW1 | BW2) & roi;

% --- Cleanup ---
BW = imclose(BW, strel('disk', closeR));
BW = imfill(BW,'holes');
BW = bwareaopen(BW, minAreaPx);

% --- Measure ---
CC = bwconncomp(BW);
S  = regionprops(CC, 'Area','Centroid','BoundingBox', ...
                     'EquivDiameter','MajorAxisLength','MinorAxisLength');

T = struct2table(S);
if ~isempty(T)
    T.CentroidX_px = T.Centroid(:,1);  T.CentroidY_px = T.Centroid(:,2);
    T.CentroidX_um = T.CentroidX_px * pixelSize_um;
    T.CentroidY_um = T.CentroidY_px * pixelSize_um;
    T.EquivDiameter_um   = T.EquivDiameter   * pixelSize_um;
    T.MajorAxisLength_um = T.MajorAxisLength * pixelSize_um;
    T.MinorAxisLength_um = T.MinorAxisLength * pixelSize_um;
    T.Area_um2           = T.Area * pixelSize_um^2;
end
toc,
% --- Plots (see each step) ---
figure('Name','Robust dark-blob detection');
tiledlayout(3,4,'Padding','compact','TileSpacing','compact');
nexttile; imshow(I0,[]); title('1) Original');
nexttile; imshow(bg,[]);  title('2) Background');
nexttile; imshow(F,[]);   title('3) Flat-fielded');

nexttile; imshow(Zd,[]);  title('4) Local z-score (dark↑)');
nexttile; imshow(BH,[]);  title('5) Black-hat (dark blobs)');
nexttile; imshow(LOGmax,[]); title('6) Max LoG (multi-scale)');

nexttile; imshow(score,[]); title('7) Combined score');
nexttile; imshow(BW1,[]); title(sprintf('8) Score≥%d%%', 100-darkPercent));
nexttile; imshow(BW2 & roi,[]); title(sprintf('9) Adaptive (sens=%.2f)', adaptSens));
nexttile; imshow(BW,[]); title('10) Union + cleanup');

% Overlay detections
nexttile; imshow(I0,[]); hold on;
for k=1:height(T)
    rectangle('Position', T.BoundingBox(k,:), 'EdgeColor','r','LineWidth',1.5);
    plot(T.CentroidX_px(k), T.CentroidY_px(k), 'g+','MarkerSize',7,'LineWidth',1.2);
end
title(sprintf('11) Detections: %d', height(T))); hold off;

% Simple “confidence” map = score masked by BW
conf = score .* BW;
nexttile; imshow(conf,[]); title('12) Masked score (confidence)');

% Stats + export
fprintf('Dark blobs detected: %d\n', height(T));
if ~isempty(T)
    fprintf('Median equiv. diam = %.2f px (%.2f µm)\n', ...
        median(T.EquivDiameter), median(T.EquivDiameter_um));
    writetable(T, 'crystal_measurements_robust.csv');
    disp('Saved: crystal_measurements_robust.csv');
end

end
