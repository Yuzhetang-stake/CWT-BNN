function yhat = predict_bnn(model, F)
% PREDICT_BNN
% model.net: trained network
% F: N x P features
Xnet = F';
yhat = model.net(Xnet)';
end
