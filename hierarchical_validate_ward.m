function T = hierarchical_validate_ward(X, c_range, varargin)
% HIERARCHICAL_VALIDATE_WARD
% Validation of Ward hierarchical clustering with:
%   - Silhouette coefficient (↑ better)
%   - Davies–Bouldin Index (↓ better)
%   - Jaccard stability via Hungarian matching (↑ better)
%
% INPUT
%   X        : n x d data (already reduced, e.g. 4 PCs)
%   c_range  : array of cluster numbers, e.g. 2:6
%
% OPTIONS (name-value)
%   'B'             : 100      % subsample/ bootstrap runs
%   'SubsampleRate' : 0.8      % fraction of points per run
%   'WithReplacement' : false  % true = bootstrap, false = subsample
%   'Verbose'       : true
%
% OUTPUT
%   T : table with columns [c, SC, DBI, Jaccard_mean, Jaccard_std]

p = inputParser;
addParameter(p,'B',100);
addParameter(p,'SubsampleRate',0.8);
addParameter(p,'WithReplacement',false);
addParameter(p,'Verbose',true);
parse(p,varargin{:});
B = p.Results.B;
q = p.Results.SubsampleRate;
withRep = p.Results.WithReplacement;
vb   = p.Results.Verbose;

if exist('matchpairs','file')~=2
    error('Optimization Toolbox required: matchpairs (Hungarian).');
end

rng('default');
n = size(X,1);
C = numel(c_range);
SC = nan(C,1); DBI = nan(C,1); Jm = nan(C,1); Js = nan(C,1);

for t = 1:C
    c = c_range(t);
    if vb, fprintf('\n=== Ward HC: c = %d ===\n', c); end

    % ---- reference clustering (Ward linkage) ----
    Z = linkage(X,'ward');
    ref = cluster(Z,'maxclust',c);

    % ---- internal validity ----
    SC(t)  = mean(silhouette(X, ref));
    DBI(t) = evalclusters(X, ref, 'DaviesBouldin').CriterionValues;

    % ---- stability via resampling ----
    Jruns = zeros(B,1);
    for b = 1:B
        m = max(2, round(q*n));
        if withRep
            I = randi(n, m, 1);   % bootstrap
        else
            I = randperm(n, m)';  % subsample
        end

        % recluster on the subset
        Zsub = linkage(X(I,:),'ward');
        sub = cluster(Zsub,'maxclust',c);

        % overlap matrix between ref(I) and sub
        refI = ref(I);
        M = zeros(c,c);
        for i = 1:c
            Ii = (refI==i);
            for j = 1:c
                Ij = (sub==j);
                M(i,j) = sum(Ii & Ij);
            end
        end

        % Hungarian
        cost = max(M(:)) - M;
        costUnmatched = max(cost(:)) + 1;
        pairs = matchpairs(cost, costUnmatched);

        % Jaccard
        Jc = zeros(c,1);
        for rpair = 1:size(pairs,1)
            i = pairs(rpair,1); j = pairs(rpair,2);
            A = find(refI==i);
            Bset = find(sub==j);
            inter = numel(intersect(A,Bset));
            uni   = numel(union(A,Bset));
            Jc(rpair) = inter / max(1,uni);
        end
        Jruns(b) = mean(Jc);
    end
    Jm(t) = mean(Jruns); Js(t) = std(Jruns);

    if vb
        fprintf('SC=%.3f  DBI=%.3f  Jaccard=%.3f ± %.3f\n', SC(t), DBI(t), Jm(t), Js(t));
    end
end

% ---- results table ----
T = table(c_range(:), SC, DBI, Jm, Js, ...
          'VariableNames', {'c','SC','DBI','Jaccard_mean','Jaccard_std'});

% ---- plots ----
figure('Color','w'); tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
nexttile; plot(c_range, SC,'-o'); xlabel('c'); ylabel('Silhouette (↑)'); grid on;
nexttile; plot(c_range, DBI,'-o'); xlabel('c'); ylabel('DBI (↓)'); grid on;
nexttile; errorbar(c_range, Jm, Js,'-o'); xlabel('c'); ylabel('Jaccard stability (↑)'); grid on;
sgtitle('Hierarchical clustering (Ward) validation');
end