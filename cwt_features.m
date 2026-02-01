function F = cwt_features(X, wl, params)
% CWT_FEATURES
% X: N x B reflectance
% wl: 1 x B wavelength vector
% params.wavelet: e.g., "amor"
% params.scales: vector of scales
% params.stat: feature statistic per scale: "energy" | "meanabs" | "maxabs"

arguments
    X double
    wl double
    params.wavelet string = "amor"
    params.scales double = 1:32
    params.stat string   = "energy"
end

N = size(X,1);
S = numel(params.scales);

F = zeros(N, S);

% Use sample spacing based on wavelength step
dw = mean(diff(wl));

for i = 1:N
    sig = X(i,:);
    % Continuous wavelet transform
    % cfs: S x B (depending on scales)
    cfs = cwt(sig, params.scales, params.wavelet, "SamplingPeriod", dw);

    % Summarize coefficients per scale
    switch lower(params.stat)
        case "energy"
            F(i,:) = sum(abs(cfs).^2, 2)';      % energy per scale
        case "meanabs"
            F(i,:) = mean(abs(cfs), 2)';        % mean abs per scale
        case "maxabs"
            F(i,:) = max(abs(cfs), [], 2)';     % max abs per scale
        otherwise
            error("Unknown params.stat: %s", params.stat);
    end
end

% Optional: feature normalization
F = zscore(F);

end
