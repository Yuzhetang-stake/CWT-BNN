function demo_cwt_bnn()
% DEMO_CWT_BNN
% Minimal runnable demo: CWT feature extraction + Bayesian NN regression.
% Replace the example data loader with your real dataset.

clc; clear; close all;

% -------------------------
% 0) Reproducibility
% -------------------------
rng(42, "twister");

% -------------------------
% 1) Load example data
% -------------------------
% Example format:
% X: N x B reflectance
% wl: 1 x B wavelengths
% y: N x 1 target (e.g., LNC or LDM)

% ---- Replace this section with your own loader ----
N = 300;
B = 200;
wl = linspace(350, 2500, B);
X  = 0.2 + 0.05*randn(N, B);     % dummy reflectance
y  = 2.0 + 0.3*randn(N, 1);      % dummy target (e.g., LNC)
% ---------------------------------------------------

% -------------------------
% 2) Split data
% -------------------------
idx = randperm(N);
nTrain = round(0.7*N);
nVal   = round(0.15*N);

iTrain = idx(1:nTrain);
iVal   = idx(nTrain+1:nTrain+nVal);
iTest  = idx(nTrain+nVal+1:end);

Xtr = X(iTrain,:); ytr = y(iTrain);
Xva = X(iVal,:);   yva = y(iVal);
Xte = X(iTest,:);  yte = y(iTest);

% -------------------------
% 3) CWT feature extraction
% -------------------------
paramsCWT.wavelet = "amor";   % Morlet (analytic)
paramsCWT.scales  = 1:32;     % example scales
paramsCWT.stat    = "energy"; % feature summarization: energy per scale

Ftr = cwt_features(Xtr, wl, paramsCWT);
Fva = cwt_features(Xva, wl, paramsCWT);
Fte = cwt_features(Xte, wl, paramsCWT);

% -------------------------
% 4) Train Bayesian NN
% -------------------------
paramsNN.hiddenSizes = [20 10]; % you can tune
paramsNN.trainFcn    = "trainbr"; % Bayesian regularization
paramsNN.maxEpochs   = 2000;

model = train_bnn(Ftr, ytr, paramsNN);

% -------------------------
% 5) Predict & evaluate
% -------------------------
yhat_va = predict_bnn(model, Fva);
yhat_te = predict_bnn(model, Fte);

[rmse_va, r2_va] = eval_metrics(yva, yhat_va);
[rmse_te, r2_te] = eval_metrics(yte, yhat_te);

fprintf("Validation: RMSE = %.4f, R2 = %.4f\n", rmse_va, r2_va);
fprintf("Test:       RMSE = %.4f, R2 = %.4f\n", rmse_te, r2_te);

% -------------------------
% 6) Plot
% -------------------------
figure;
scatter(yte, yhat_te, "filled");
xlabel("Observed"); ylabel("Predicted");
title(sprintf("CWT-BNN (Test): R^2=%.3f, RMSE=%.3f", r2_te, rmse_te));
grid on;

end

function [rmse, r2] = eval_metrics(y, yhat)
y = y(:); yhat = yhat(:);
rmse = sqrt(mean((y - yhat).^2));
ssRes = sum((y - yhat).^2);
ssTot = sum((y - mean(y)).^2) + eps;
r2 = 1 - ssRes/ssTot;
end
