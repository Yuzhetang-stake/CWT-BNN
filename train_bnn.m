function model = train_bnn(F, y, params)
% TRAIN_BNN
% F: N x P features
% y: N x 1 target
% params.hiddenSizes: e.g., [20 10]
% params.trainFcn: "trainbr" (Bayesian regularization)
% params.maxEpochs: e.g., 2000

arguments
    F double
    y double
    params.hiddenSizes double = 20
    params.trainFcn string = "trainbr"
    params.maxEpochs double = 2000
end

% fitnet expects inputs as P x N and targets as 1 x N
Xnet = F';
Ynet = y';

net = fitnet(params.hiddenSizes, char(params.trainFcn));
net.trainParam.epochs = params.maxEpochs;

% Use all input as "train" inside this function; user controls split outside
net.divideFcn = "dividetrain";

% Train
[net, tr] = train(net, Xnet, Ynet);

model.net = net;
model.tr  = tr;

end
