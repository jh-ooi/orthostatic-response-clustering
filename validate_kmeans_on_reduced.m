function T = validate_kmeans_on_reduced(X, k_range, varargin)
% VALIDATE_KMEANS_ON_REDUCED
% Assumes X is already reduced (e.g., PCA or UMAP). For each k in k_range:
%   - runs k-means (robust settings)
%   - records WCSS (for elbow), Silhouette (SC), Davies–Bouldin (DBI)
%   - computes Jaccard stability (mean ± std) via Hungarian matching
%
% INPUT
%   X        : n x d (already-reduced data; rows=samples)
%   k_range  : vector of candidate k, e.g. 2:8
%
% OPTIONS (name-value)
%   'Replicates'      : 50
%   'MaxIter'         : 1000
%   'B'               : 100          % subsample/bootstrap runs
%   'SubsampleRate'   : 0.80         % fraction of samples per run
%   'WithReplacement' : false        % true=bootstrap, false=subsample
%   'Verbose'         : true
%
% OUTPUT
%   T : table with columns:
%       k, WCSS, SC, DBI, Jaccard_mean, Jaccard_std

% ---- params ----
p = inputParser;
addParameter(p,'Replicates',50);
addParameter(p,'MaxIter',1000);
addParameter(p,'B',100);
addParameter(p,'SubsampleRate',0.80);
addParameter(p,'WithReplacement',false);
addParameter(p,'Verbose',true);
parse(p,varargin{:});
rpts = p.Results.Replicates;
mxit = p.Results.MaxIter;
B    = p.Results.B;
q    = p.Results.SubsampleRate;
withRep = p.Results.WithReplacement;
vb   = p.Results.Verbose;

if exist('matchpairs','file')~=2
    error('Optimization Toolbox required: matchpairs (Hungarian).');
end

rng('default'); n = size(X,1);
K = numel(k_range);
WCSS = nan(K,1); SC = nan(K,1); DBI = nan(K,1); Jm = nan(K,1); Js = nan(K,1);

for t = 1:K
    k = k_range(t);
    if vb, fprintf('\n=== k = %d ===\n', k); end

    % --- k-means on ALL data (robust init) ---
    [lbl, C, sumd] = kmeans(X, k, ...
        'Replicates', rpts, 'MaxIter', mxit, ...
        'Distance','sqeuclidean', 'Display','off');

    % WCSS (elbow)
    WCSS(t) = sum(sumd);

    % Internal validity in THIS space
    SC(t)  = mean(silhouette(X, lbl));
    DBI(t) = evalclusters(X, lbl, 'DaviesBouldin').CriterionValues;

    % --- Jaccard stability via subsampling + Hungarian ---
    Jruns = zeros(B,1);
    for b = 1:B
        m = max(2, round(q*n));

        if withRep
            I = randi(n,m,1);
        else
            I = randperm(n,m)';
        end
        
        lbl_sub = kmeans(X(I,:), k, ...
            'Replicates', max(5,ceil(rpts/5)), 'MaxIter', mxit, ...
            'Distance','sqeuclidean', 'Display','off');


        % overlap matrix M(i,j) between reference labels (lbl(I)) and lbl_sub
        refI = lbl(I); M = zeros(k,k);
        for i = 1:k
            Ii = (refI==i);
            for j = 1:k
                Ij = (lbl_sub==j);
                M(i,j) = sum(Ii & Ij);
            end
        end

        % Hungarian: maximize overlap by minimizing positive costs
        cost = max(M(:)) - M;                   % >=0, larger overlap -> smaller cost
        costUnmatched = max(cost(:)) + 1;       % finite, larger than any real match
        pairs = matchpairs(cost, costUnmatched); % [refIdx, subIdx]

        % Jaccard per matched pair, then average
        Jc = zeros(k,1);
        for rpair = 1:size(pairs,1)
            i = pairs(rpair,1); j = pairs(rpair,2);
            A = find(refI==i);
            Bset = find(lbl_sub==j);
            inter = numel(intersect(A,Bset));
            uni   = numel(union(A,Bset));
            Jc(rpair) = inter / max(1,uni);
        end
        Jruns(b) = mean(Jc);
    end
    Jm(t) = mean(Jruns); Js(t) = std(Jruns);

    if vb
        fprintf('WCSS=%g  SC=%.3f  DBI=%.3f  Jaccard=%.3f ± %.3f\n', ...
                 WCSS(t), SC(t), DBI(t), Jm(t), Js(t));
    end
end

% Results table
T = table(k_range(:), WCSS, SC, DBI, Jm, Js, ...
    'VariableNames', {'k','WCSS','SC','DBI','Jaccard_mean','Jaccard_std'});

% === quick plots ===
figure('Color','w'); tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

% Elbow
nexttile; plot(k_range, WCSS, '-o'); xlabel('k'); ylabel('WCSS (↓)'); 
title('Elbow (inertia)'); grid on;

% Silhouette
nexttile; plot(k_range, SC, '-o'); xlabel('k'); ylabel('Silhouette (↑)'); 
title('Silhouette'); grid on;

% DBI
nexttile; plot(k_range, DBI, '-o'); xlabel('k'); ylabel('DBI (↓)');
title('Davies–Bouldin'); grid on;

% Jaccard stability
nexttile; errorbar(k_range, Jm, Js, '-o'); xlabel('k'); ylabel('Jaccard stability (↑)');
title('Stability (subsample + Hungarian)'); grid on;

sgtitle('k-means validation on reduced space');

end
