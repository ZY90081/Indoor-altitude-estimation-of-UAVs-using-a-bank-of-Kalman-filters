% MMAE approach to estimate altitude of drone
% Liu Yang @ Stony Brook
% 08 - 21 - 2019

% Reference: 

% ==============================================================

clc;
close all;
clear all;

% load simulated data.
load('x.mat');  % ground truth
load('z.mat');  % observation
load('a.mat');  % obstacles

% load real data.
% load('zero_order_arraytest-2-1.bag.mat')
% z = [iu;id];
% N = size(z,2);
% x = [1.7*ones(1,N);zeros(1,N)];
% a = [2.88*ones(1,N); zeros(1,N)];


% Parameters.
Ts = 0.02;      % sampling interval
Ho = 3;         % height of room
alpha = 0.8;    % forgetting factor

N = size(x,2);  % number of time steps
Dx = size(x,1)+size(a,1);  % dimension of state
Dy = size(z,1);  % dimension of observation
M = 4;          % number of models
y = [z(1,:)-Ho*ones(1,N);z(2,:)]; % modified observations.
clear z;

sig_u = 0.1;  % variance of noise on acceleration
sig_v = 0.001;  % variance of noise on observation
sig_a = 1;      % variance of noise on obstacles
 
% Models.
F = [1 Ts 0 0;0 1 0 0;0 0 0 0;0 0 0 0];  % transition matrix
H(:,:,1) = [-1 0 0 0; 1 0 0 0];
H(:,:,2) = [-1 0 -1 0;1 0 0 0];
H(:,:,3) = [-1 0 0 0;1 0 0 -1];
H(:,:,4) = [-1 0 -1 0;1 0 0 -1];  % observation matrices
Q = [Ts^4/4*sig_u Ts^4/2*sig_u 0 0;Ts^4/2*sig_u Ts^4*sig_u 0 0;0 0 sig_a 0;0 0 0 sig_a];
R = eye(Dy)*sig_v;

% Initialization
X = zeros(Dx,M,N);
P = repmat(eye(Dx),1,1,M);
L = zeros(M,N);
tempx = zeros(Dx,1);
tempP = eye(Dx);
loglike = zeros(M,1);
xhat = zeros(Dx,N);

% MMAE approach (MMKF)
for i = 1:N
    for m = 1:M
        
        tempx_ = F*tempx;
        tempP_ = F*tempP*F' + Q;
        
        z = y(:,i) - H(:,:,m)*tempx_; % innovation
        S = H(:,:,m)*tempP_*H(:,:,m)' + R;
        K = tempP_*H(:,:,m)'*inv(S);
        
        x_ = tempx_ + K*z;
        p_ = tempP_ - K*H(:,:,m)*tempP_;
        
        X(:,m,i) = x_;
        P(:,:,m) = p_;
        
        loglike(m) = alpha*loglike(m) - 0.5*(z'*inv(S)*z+log(det(S))+Dy*log(2*pi)); 
    end
    % transfer log-likelihood to likelihood
    L(:,i) = exp(loglike-max(loglike)*ones(M,1));
    L(:,i) = L(:,i)./sum(L(:,i)); % normalization
    xhat(:,i) = X(:,:,i)*L(:,i); % estimated state
    for m = 1:M
        Pcor(:,:,m) = (X(:,m,i)-xhat(:,i))*(X(:,m,i)-xhat(:,i))';
    end
    phat = sum(( P + Pcor ).*reshape(L(:,i),[1 1 M]),3); % estimated cov
    
    tempx = xhat(:,i); % set for next iteration
    tempP = phat; 

end

figure()
plot(Ts:Ts:Ts*N,a(1,:),'k-','LineWidth', 4);
hold on; grid on;
plot(Ts:Ts:Ts*N,x(1,:),'b-','LineWidth', 4);
plot(Ts:Ts:Ts*N,Ho-xhat(3,:),'m-.','LineWidth', 2);
plot(Ts:Ts:Ts*N,xhat(1,:),'r-.','LineWidth', 3);
plot(Ts:Ts:Ts*N,a(2,:),'k-','LineWidth', 4);
plot(Ts:Ts:Ts*N,xhat(4,:),'m-.','LineWidth', 2);
ylim([0,3]);
xlabel('Time(s)');
ylabel('Altitude(m)');
legend('Obstacles on Ceiling and floor','True trajectory','Estimated obstacles','Estimated trajectory');


figure()
subplot(4,1,1)
plot(Ts:Ts:Ts*N,L(1,:),'LineWidth', 2);
title('No obstacles');
subplot(4,1,2)
plot(Ts:Ts:Ts*N,L(2,:),'LineWidth', 2);
title('Obstacles above');
subplot(4,1,3)
plot(Ts:Ts:Ts*N,L(3,:),'LineWidth', 2);
title('Obstacles below');
ylabel('Weights of models');
subplot(4,1,4)
plot(Ts:Ts:Ts*N,L(4,:),'LineWidth', 2);
title('Obstacles above and below');
xlabel('Time (s)');

MSE = mean((xhat(1,:)-x(1,:)).^2)

