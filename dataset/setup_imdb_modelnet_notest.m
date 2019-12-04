function imdb = setup_imdb_modelnet_notest(datasetDir, varargin)

opts.seed = 0 ;             % random seed generator
opts.ratio = [0.8 0.2];     % train:val ratio
opts.ext = '.jpg';          % extension of target files
opts.extmesh = '.off';      % extension of target mesh files
opts = vl_argparse(opts, varargin);

opts.ratio = opts.ratio(1:2)/sum(opts.ratio(1:2));
rng(opts.seed);
imdb.imageDir = datasetDir;

% meta
folders = {};
fprintf('Scanning for classes ... ');
contents = dir(imdb.imageDir);
for i=1:numel(contents),
    if contents(i).isdir, folders = [folders contents(i).name]; end
end
imdb.meta.classes = setdiff(folders,{'.','..'});
imdb.meta.sets = {'train', 'val'};
fprintf('%d classes found! \n', length(imdb.meta.classes));

% images
imdb.images.name    = {};
imdb.images.class   = [];
imdb.images.set     = [];
imdb.images.sid     = [];
fprintf('Scanning for images: \n');
[imdb, nTrainShapes] = scan_images(imdb, opts);
if nTrainShapes == 0
    fprintf('No images found. Render them using create_data function!\n')
end

[imdb.images.sid, I] = sort(imdb.images.sid);
imdb.images.name = imdb.images.name(I);
imdb.images.class = imdb.images.class(I);
imdb.images.set = imdb.images.set(I);

nViews = sum(imdb.images.sid==1); 
imdb.images.id = 1:length(imdb.images.name);

end

function [imdb, nTrainShapes] = scan_images(imdb, opts)
for ci = 1:length(imdb.meta.classes),
    fprintf('  [%2d/%2d] %s ... ', ci, length(imdb.meta.classes), ...
        imdb.meta.classes{ci});
    trainDir = fullfile(imdb.imageDir,imdb.meta.classes{ci});
    valDir = fullfile(imdb.imageDir,imdb.meta.classes{ci});
    
    % train
    
        files = dir(fullfile(trainDir, ['*' opts.ext]));
        fileNames = {files.name};
        nTrainImages = length(fileNames);
        imVids = cellfun(@(s) get_shape_vid(s), fileNames);
        [~,I] = sort(imVids);
        fileNames = fileNames(I); % order images wrt view id
        sNames = cellfun(@(s) get_shape_name(s), fileNames, ...
            'UniformOutput', false);
        sNamesUniq = unique(sNames);
        nTrainShapes = length(sNamesUniq);
        [~,imSids] = ismember(sNames,sNamesUniq);
        if isempty(imdb.images.sid), maxSid = 0;
        else maxSid = max(imdb.images.sid); end
        imdb.images.sid = [imdb.images.sid maxSid+imSids];
        imdb.images.set = [imdb.images.set ones(1,nTrainImages)];
        imdb.images.class = [imdb.images.class ci*ones(1,nTrainImages)];
        imdb.images.name = [imdb.images.name ...
            cellfun(@(s) fullfile(imdb.meta.classes{ci},s), ...
            fileNames, 'UniformOutput',false)];
    
    % val

    if opts.ratio(2)>0,
        inds = (imdb.images.set==1 & imdb.images.class==ci);
        trainvalSids = unique(imdb.images.sid(inds));
        nValShapes = floor(opts.ratio(2)*numel(trainvalSids));
        trainvalSids = trainvalSids(randperm(numel(trainvalSids)));
        valSids = trainvalSids(1:nValShapes);
        inds = ismember(imdb.images.sid,valSids);
        imdb.images.set(inds) = 2;
        nTrainShapes = nTrainShapes - nValShapes;
    else
        nValShapes = 0;
    end
        
    fprintf('\ttrain/val: %d/%d (shapes)\n', ...
        nTrainShapes, nValShapes);
end
end

function shapename = get_shape_name(filename)
suffix_idx = strfind(filename,'_');
if isempty(suffix_idx),
    shapename = [];
else
    shapename = filename(1:suffix_idx(end)-1);
end
end

function vid = get_shape_vid(filename)
suffix_idx = strfind(filename,'_');
ext_idx = strfind(filename,'.');
if isempty(suffix_idx) || isempty(ext_idx),
    vid = [];
else
    vid = str2double(filename(suffix_idx(end)+1:ext_idx(end)-1));
end
end
