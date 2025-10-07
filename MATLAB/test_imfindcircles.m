close all,
%% Detect circular dots with imfindcircles
% File: c2w4_LaB6_20xmag.jpg
I = imread('image_2_prakriti_cropped.png');     % <-- put your path if different
if size(I,3)>1, I = rgb2gray(I); end
I = im2single(I);

% --- Preprocess: remove vignette & boost local contrast
bg  = imgaussfilt(I, 40);               % large sigma -> smooth background
J   = I - bg;                           % flat-field (zero-mean-ish)
J   = mat2gray(J);                      % normalize to [0,1]
J   = adapthisteq(J,'NumTiles',[8 8]);  % local contrast (CLAHE)
J   = imbilatfilt(J, 0.1, 3);           % denoise but keep edges

% --- Pick a radius range (in pixels).
rmin = 3;    
rmax = 30;   % allow larger circles (since we will filter afterwards)

% --- Try both polarities
[centDark, rDark, mDark] = imfindcircles(J, [rmin rmax], ...
    'ObjectPolarity','dark', 'Sensitivity',0.92, 'EdgeThreshold',0.05, ...
    'Method','PhaseCode');

[centBright, rBright, mBright] = imfindcircles(J, [rmin rmax], ...
    'ObjectPolarity','bright', 'Sensitivity',0.92, 'EdgeThreshold',0.05, ...
    'Method','PhaseCode');

% --- Merge detections
cent  = [centDark;   centBright];
radii = [rDark;      rBright];
metric= [mDark;      mBright];

if ~isempty(cent)
    % suppress near-duplicate centers (3 px tolerance)
    idx = true(size(radii));
    for i = 1:numel(radii)
        if ~idx(i), continue; end
        d = vecnorm(cent - cent(i,:), 2, 2);
        dup = find(d>0 & d<=3);
        if ~isempty(dup)
            vals   = [metric(i); metric(dup)];
            idxAll = [i; dup];
            [~,k]  = max(vals);
            keepIdx = idxAll(k);
            idx([i; dup]) = false;   % drop all…
            idx(keepIdx)  = true;    % …except best
        end
    end
    cent  = cent(idx,:);  
    radii = radii(idx);  
    metric= metric(idx);
end

% --- NEW STEP: only keep circles with radius > 10 px
bigIdx = radii > 10;
cent   = cent(bigIdx,:);
radii  = radii(bigIdx);
metric = metric(bigIdx);

% --- Visualize
figure; imshow(I,[]); title(sprintf('Detections (r>10px): %d', numel(radii))); hold on;
viscircles(cent, radii, 'LineWidth',0.8,'Color','r');
hold off;

% --- Size statistics
fprintf('Detected %d circles with radius > 10 pixels\n', numel(radii));
if ~isempty(radii)
    figure; histogram(radii, 'BinMethod','sturges');
    xlabel('Radius (pixels)'); ylabel('Count'); title('Radius distribution (r > 10px)');
end
