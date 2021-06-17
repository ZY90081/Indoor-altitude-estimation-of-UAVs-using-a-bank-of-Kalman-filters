% Generate simple scenarios of altitude estimation for algorithm testing
% Liu Yang @ Stony Brook
% 03 - 27 - 2019

% Reference:

% ==============================================================

clc;
close all;
clear all;

% Parameters.
Ts = 0.02;      % sampling interval
H = 3;          % height of room
ho = 1.5;       % initial height of drone
hodot = 0;      % initial vertical velocity
sig_u = 0.001;  % variance of noise on acceleration
sig_v = 0.001;  % variance of noise on observation
N = 2000;       % number of samples

% Scenario.
T = Ts:Ts:Ts*N; % time axis
au = zeros(1,N);% obstacle up
ad = zeros(1,N);% obstacle down

ad(201:300) = 0.3*ones(1,100);
au(301:400) = 0.3*ones(1,100);    % set square obstacles

ad(500:600) = 0:(1/100):1;  
au(600:700) = flip(0:(1/100):1);  % set wedge obstacles

ad(801:1000) = 0.1+0.6*sin(0.015*(1:200));  
au(1001:1200) = 0.1-0.6*cos(0.015*(101:300));  % set obstacles

ad(1401:1600) = 0.1*sin(0.05*(1:200))+0.3*sin(0.01*(1:200));  
au(1401:1600) = -0.1*sin(0.05*(101:300))+0.8*sin(0.01*(1:200));  % set random obstacles up and down.

a = [H-au;ad];     % vector of obstacles.

% Generate trajectory of drone
h = zeros(1,N);    % true height of drone
hdot = zeros(1,N); % true vertical velocity of drone
x = [h;hdot];      % state of drone
Gx = [1 Ts;0 1];   % matrices of state equation
Gu = [0.5*Ts^2;Ts];

while sum((x(1,:)<=ad)|(x(1,:)>=H-au))>=1    % constraint
    for i = 1:N
        if i == 1
            x(:,i) = Gx*[ho;hodot]+Gu.*randn(2,1)*sqrt(sig_u);
        else
            x(:,i) = Gx*x(:,i-1)+Gu.*randn(2,1)*sqrt(sig_u);      
        end
    end
end

% plot scenario.
figure()
plot(T,ad,'-k','LineWidth',2); hold on;
plot(T,H-au,'-k','LineWidth',2);
xlabel('time(s)');
ylabel('height(m)');
plot(T,x(1,:),'--b','LineWidth',2);


% Mesurements.
z = zeros(2,N);    % measurements of two infrared sensors (up and down)
Fx = [-1 0;1 0];   % matrices of observation equations
Fa = [1 0;0 -1];
for i=1:N
    z(:,i) = Fx*x(:,i)+Fa*a(:,i)+randn(2,1)*sqrt(sig_v);
end

figure()
plot(T,z(1,:),'LineWidth',2); hold on;
plot(T,z(2,:),'LineWidth',2);
legend('infrared up','infrared down');
xlabel('time(s)');
ylabel('height(m)');

save z z;  % observation 
save x x;  % trojectory 
save a a;  % obstacles

