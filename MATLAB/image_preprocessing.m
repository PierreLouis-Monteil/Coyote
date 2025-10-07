
function detect_dark_on_flatfield(imgPath, pixelSize_um)
% Detect darkest blobs, output centroid and dimensions
% Usage:
%   detect_dark_on_flatfield('image.png', 0.5)
%   where pixelSize_um is the size of one pixel (e.g. 0.5 µm/px)

tic;

if nargin < 1
    imgPath = 'image_2_prakriti_cropped.png';
end
if nargin < 2
    pixelSize_um = 1;  % default: report in pixels if scale unknown
end

% ---- Tunables ----
bgSigma     = 40;   % Gaussian sigma for background (vignette)
darkPercent = 12;   % keep darkest X% pixels
minAreaPx   = 120;  % remove blobs smaller than this
closeR      = 2;    % morphological closing radius
rimFrac     = 0.92; % ignore vignetted rim
% ------------------

% Read & grayscale
I0 = imread(imgPath);
if size(I0,3)>1, I0 = rgb2gray(I0); end
I0 = im2single(I0);

% --- Flat-field only ---
bg = imgaussfilt(I0, bgSigma);
F  = I0 - bg;                     % flat-fielded
F  = F - min(F(:));
F  = F ./ max(F(:)+eps);          % robuslty normalize to [0,1]

% --- Threshold darkest part of FLAT-FIELD only ---
th = prctile(F(:), darkPercent);
BW = F <= th;



% Optional: ignore vignette rim
[h,w] = size(F);
[xg,yg] = meshgrid((1:w)-w/2,(1:h)-h/2);
roi = sqrt(xg.^2 + yg.^2) <= min(h,w)*0.5*rimFrac;
BW(~roi) = 0;

% Clean mask
BW = imclose(BW, strel('disk', closeR));
BW = imfill(BW,'holes');
BW = bwareaopen(BW, minAreaPx);

% --- Label & measure properties ---
CC = bwconncomp(BW); %Finds all connected groups of “1” pixels in BW
S  = regionprops(CC, 'Area','Centroid','BoundingBox','EquivDiameter', ...
                      'MajorAxisLength','MinorAxisLength');

% Convert to table with pixel + micron units
T = struct2table(S);
if ~isempty(T)
    T.CentroidX_px = T.Centroid(:,1);
    T.CentroidY_px = T.Centroid(:,2);
    T.CentroidX_um = T.CentroidX_px * pixelSize_um;
    T.CentroidY_um = T.CentroidY_px * pixelSize_um;
    T.EquivDiameter_um   = T.EquivDiameter * pixelSize_um;
    T.MajorAxisLength_um = T.MajorAxisLength * pixelSize_um;
    T.MinorAxisLength_um = T.MinorAxisLength * pixelSize_um;
    T.Area_um2           = T.Area * (pixelSize_um^2);
end
toc;
% ---- Plots ----
figure('Name','Flat-field detection','Position',[80 80 1200 600]);
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
nexttile; imshow(I0,[]); title('1) Original');
nexttile; imshow(bg,[]); title('2) Estimated background');
nexttile; imshow(F,[]);  title('3) Flat-fielded');
nexttile; imshow(BW);    title(sprintf('4) Darkest %d%% + cleanup',darkPercent));

% Overlay detections
nexttile; imshow(I0,[]); hold on;
for k=1:height(T)
    rectangle('Position', T.BoundingBox(k,:), 'EdgeColor','r','LineWidth',1.5);
    plot(T.CentroidX_px(k), T.CentroidY_px(k), 'g+','MarkerSize',7,'LineWidth',1.2);
end
title(sprintf('5) Detections: %d', height(T))); hold off;

% Darkness score for intuition
score = (th - F) .* BW; score(score<0)=0;
nexttile; imshow(score,[]); title('6) Darkness score (masked)');

% ---- Stats in console ----
fprintf('Dark spots detected: %d\n', height(T));
if ~isempty(T)
    fprintf('Median area = %.1f px² (%.2f µm²)\n', median(T.Area), median(T.Area_um2));
    fprintf('Median diameter = %.2f px (%.2f µm)\n', median(T.EquivDiameter), median(T.EquivDiameter_um));
end

% ---- Export centroids + dimensions ----
if ~isempty(T)
    disp('Crystal positions and dimensions:');
    disp(T(:, {'CentroidX_um','CentroidY_um','EquivDiameter_um','MajorAxisLength_um','MinorAxisLength_um','Area_um2'}));
    writetable(T,'crystal_measurements.csv');
    fprintf('Results exported to crystal_measurements.csv\n');
end


end
