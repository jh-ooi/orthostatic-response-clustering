function T = fuzzy_validate_on_reduced(X, c_range, varargin)
% Fuzzy c-means validation on already-reduced data (rows = samples).
% Computes: PC (↑), PE (↓), XB (↓), plus SC (↑), DBI (↓) from hardened labels,
% and Jaccard stability (↑) via Hungarian matching under subsampling.

% ---- params ----
p = inputParser;
addParameter(p,'m',2.0);              % fuzzifier
addParameter(p,'MaxIter',100);        % FCM iterations
addParameter(p,'MinImprov',1e-5);     % FCM tolerance
addParameter(p,'B',100);              % stability resamples
addParameter(p,'SubsampleRate',0.80); % fraction of samples per resample
addParameter(p,'WithReplacement',false);
addParameter(p,'Verbose',true);
parse(p,varargin{:});
m          = p.Results.m;
maxIter    = p.Results.MaxIter;
minImprov  = p.Results.MinImprov;
B          = p.Results.B;
q          = p.Results.SubsampleRate;
withRep    = p.Results.WithReplacement;
vb         = p.Results.Verbose;

if exist('fcm','file')~=2
    error('Fuzzy Logic Toolbox required: fcm.');
end
if exist('matchpairs','file')~=2
    error('Optimization Toolbox required: matchpairs (Hungarian).');
end

rng('default');
n = size(X,1);
C = numel(c_range);
PC = nan(C,1); PE = nan(C,1); XB = nan(C,1);
SC = nan(C,1); DBI = nan(C,1);
Jm = nan(C,1); Js = nan(C,1);

for t = 1:C
    c = c_range(t);
    if vb, fprintf('\n=== FCM: c = %d (m = %.2f) ===\n', c, m); end

    % ---- fit FCM on ALL data ----
    opts = [m, maxIter, minImprov, 0];      % [m MaxIter MinImprov Display]
    [centers, U] = fcm(X, c, opts);         % centers: c x d, U: n x c (note: MATLAB's fcm returns U' in some vers.)
    % Ensure U is n x c (rows=samples). If fcm returned c x n, transpose:
    if size(U,1)~=n && size(U,2)==n, U = U'; end

    % ---- fuzzy indices ----
    PC(t) = mean(sum(U.^2, 2));             % partition coefficient
    PE(t) = -mean(sum(U .* log(U + eps), 2));
    % Xie-Beni:
    % numerator: sum_k sum_i u_ik^m * ||x_i - v_k||^2
    numXB = 0;
    for k = 1:c
        dif = X - centers(k,:);                         % n x d
        numXB = numXB + sum( (U(:,k).^m) .* sum(dif.^2,2) );
    end
    % denominator: n * min_{k!=l} ||v_k - v_l||^2
    Dc = squareform(pdist(centers));
    Dc(Dc==0) = inf;
    XB(t) = numXB / ( n * min(Dc(:)) );

    % ---- harden for SC / DBI ----
    [~, hard] = max(U, [], 2);
    SC(t)  = mean(silhouette(X, hard));
    DBI(t) = evalclusters(X, hard, 'DaviesBouldin').CriterionValues;

    % ---- stability (Jaccard) via subsampling + Hungarian on hardened labels ----
    Jruns = zeros(B,1);
    for b = 1:B
        msz = max(2, round(q*n));
        if withRep
            I = randi(n, msz, 1);
        else
            I = randperm(n, msz)'; 
        end

        % re-fit FCM on subset
        [cent_sub, Usub] = fcm(X(I,:), c, opts);
        if size(Usub,1)~=numel(I) && size(Usub,2)==numel(I), Usub = Usub'; end
        [~, hard_sub] = max(Usub, [], 2);

        % overlap matrix between reference hard(I) and hard_sub
        refI = hard(I);
        M = zeros(c,c);
        for i = 1:c
            Ii = (refI==i);
            for j = 1:c
                Ij = (hard_sub==j);
                M(i,j) = sum(Ii & Ij);
            end
        end

        % Hungarian on positive costs (maximize overlap)
        cost = max(M(:)) - M;
        costUnmatched = max(cost(:)) + 1;
        pairs = matchpairs(cost, costUnmatched);

        % Jaccard per matched pair, then average
        Jc = zeros(c,1);
        for rpair = 1:size(pairs,1)
            i = pairs(rpair,1); j = pairs(rpair,2);
            A = find(refI==i);
            Bset = find(hard_sub==j);
            inter = numel(intersect(A,Bset));
            uni   = numel(union(A,Bset));
            Jc(rpair) = inter / max(1,uni);
        end
        Jruns(b) = mean(Jc);
    end
    Jm(t) = mean(Jruns); Js(t) = std(Jruns);

    if vb
        fprintf('PC=%.3f  PE=%.3f  XB=%.3f  |  SC=%.3f  DBI=%.3f  |  Jacc=%.3f ± %.3f\n', ...
            PC(t), PE(t), XB(t), SC(t), DBI(t), Jm(t), Js(t));
    end
end

% results table
T = table(c_range(:), PC, PE, XB, SC, DBI, Jm, Js, ...
    'VariableNames', {'c','PC','PE','XB','SC','DBI','Jaccard_mean','Jaccard_std'});

% quick plots
figure('Color','w'); tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
nexttile; plot(c_range, PC,'-o'); title('Partition Coefficient (↑)'); xlabel('c'); grid on;
nexttile; plot(c_range, PE,'-o'); title('Partition Entropy (↓)'); xlabel('c'); grid on;
nexttile; plot(c_range, XB,'-o'); title('Xie–Beni (↓)'); xlabel('c'); grid on;
nexttile; plot(c_range, SC,'-o'); title('Silhouette (↑)'); xlabel('c'); grid on;
nexttile; plot(c_range, DBI,'-o'); title('Davies–Bouldin (↓)'); xlabel('c'); grid on;
nexttile; errorbar(c_range, Jm, Js, '-o'); title('Jaccard stability (↑)'); xlabel('c'); grid on;
sgtitle(sprintf('FCM validation (m=%.2f, B=%d, q=%.2f)', m, B, q));
end
