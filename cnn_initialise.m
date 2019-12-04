function net = cnn_initialise(classNames, varargin)
opts.base = 'imagenet-matconvnet-vgg-m';
opts.restart = false;
opts.nViews = 24;

opts.weightInitMethod = 'xavierimproved';
opts.scale = 1;
opts.networkType = 'dagnn'; 
opts.addBiasSamples = 1;

opts.addLossSmooth  = 1;
opts = vl_argparse(opts, varargin);

nClass = length(classNames);

% Load model
if ~ischar(opts.base),
    net = opts.base;
else
    netFilePath = fullfile('data','models', [opts.base '.mat']);
    net = load(netFilePath);
end
assert(strcmp(net.layers{end}.type, 'softmax'), 'Wrong network format');
dataTyp = class(net.layers{end-1}.weights{1});

%Initialise weights for the fc layers
widthPrev = size(net.layers{end-1}.weights{1}, 3);
nClass0 = size(net.layers{end-1}.weights{1},4);
if nClass0 ~= nClass || opts.restart,
    net.layers{end-1}.weights{1} = init_weight(opts, 1, 1, widthPrev, nClass, dataTyp);
    net.layers{end-1}.weights{2} = zeros(nClass, 1, dataTyp);
end

sz = size(net.layers{end-3}.weights{1});
net.layers{end-3}.weights{1} = init_weight(opts, sz(1), sz(2), sz(3), sz(4), dataTyp);
net.layers{end-3}.weights{2} = zeros(sz(4), 1, dataTyp);

if opts.restart,
    w_layers = find(cellfun(@(c) isfield(c,'weights'),net.layers));
    for i=w_layers(1:end-1),
        sz = size(net.layers{i}.weights{1});
        net.layers{i}.weights{1} = init_weight(opts, sz(1), sz(2), sz(3), sz(4), dataTyp);
        net.layers{i}.weights{2} = zeros(sz(4), 1, dataTyp);
    end
end
%

if opts.nViews==1
   % Swap softmax w/ softmaxloss
    net.layers{end} = struct('type', 'softmaxloss', 'name', 'loss') ;
end

% update meta data
net.meta.classes.name = classNames;
net.meta.classes.description = classNames;

if nClass==0,
    net.layers = net.layers(1:end-2);
end

if opts.nViews>1
    % convert to dagnn
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.removeLayer('prob') ;
    net.setLayerOutputs('fc8', {'prediction'}) ;
   
    relu7 = (arrayfun(@(a) strcmp(a.name, 'relu7'), net.layers)==1);
    net.addLayer('viewsa', VS('vstride', opts.nViews),net.layers(relu7).outputs{1},'xViewSa',{});
    net.addLayer('sapool', SP('vstride', opts.nViews),{net.layers(relu7).outputs{1},'xViewSa'},'xSapool',{});
    pfc8 = (arrayfun(@(a) strcmp(a.name, 'fc8'), net.layers)==1);
    psapool = (arrayfun(@(a) strcmp(a.name, 'sapool'), net.layers)==1);
    
    net.layers(pfc8).inputs{1} = 'xSapool';
    net.layers(pfc8).inputIndexes(1) = net.layers(psapool).outputIndexes(1);
    
    net.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), {'prediction','label'}, 'objective');
end
end


% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
    case 'gaussian'
        sc = 0.01/opts.scale ;
        weights = randn(h, w, in, out, type)*sc;
    case 'xavier'
        sc = sqrt(3/(h*w*in)) ;
        weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
    case 'xavierimproved'
        sc = sqrt(2/(h*w*out)) ;
        weights = randn(h, w, in, out, type)*sc ;
    otherwise
        error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end

end


% -------------------------------------------------------------------------



