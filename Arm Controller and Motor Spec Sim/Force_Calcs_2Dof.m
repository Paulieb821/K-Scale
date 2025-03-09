% Calculates the torques on J2 and J4 when applying a static horizontal
% force

% Cleanup
clear;
clc;

% System Parameters
L1 = 300/1000;          % Elbow length (m)
L2 = 400/1000;          % Forearm length (m)

% Scenario (Using absolute angles for easier mental visualization!!)
theta1 = deg2rad(45);
theta2 = deg2rad(-45);
Fh = 66;    % (N)

% Jacobian
J = [-L1*sin(theta1) -L2*sin(theta2); L1*cos(theta1) L2*cos(theta2)];
F = [Fh; 0];

% Calculate torques
T = J'*F; % Nm

fprintf("J2 requires %0.1f Nm of torque as compared to a continous limit of 20 Nm and a peak limit of 60 Nm\n", abs(T(1)))
fprintf("J4 requires %0.1f Nm of torque as compared to a continous limit of 6 Nm and a peak limit of 17 Nm\n", abs(T(2)/3))

%% 
% Define the parameters
% Define the parameters
L1 = 0.3; % Length of the first arm
L2 = 0.4; % Length of the second arm
Fh = 66;  % Force magnitude
theta1 = deg2rad(linspace(0, 90, 50));  % First joint angle (in radians)
theta2 = deg2rad(linspace(0, -90, 50)); % Second joint angle (in radians)
t1limit = 20; % Torque limit for T1
t2limit = 6;  % Torque limit for T2

% Initialize torque arrays
T1 = zeros(length(theta1), length(theta2));
T2 = zeros(length(theta1), length(theta2));

% Compute torque values
for i = 1:length(theta1)
    for j = 1:length(theta2)
        J = [-L1*sin(theta1(i)) -L2*sin(theta2(j)); 
              L1*cos(theta1(i))  L2*cos(theta2(j))];
        F = [Fh; 0];
        T = J' * F; % Torque values
        T1(i, j) = T(1); % First torque component
        T2(i, j) = T(2); % Second torque component
    end
end

% Create meshgrid for plotting
[Theta1, Theta2] = meshgrid(theta1, theta2);

figure;

% Surface plot for T1
subplot(1, 2, 1);
surf(rad2deg(Theta1), rad2deg(Theta2), -T1');
title('Torque T1');
xlabel('\theta_1 (deg)');
ylabel('\theta_2 (deg)');
zlabel('T1 (Nm)');
grid on;
hold on;
% Add torque limit plane for T1
Z1_limit = t1limit * ones(size(Theta1));
surf(rad2deg(Theta1), rad2deg(Theta2), Z1_limit', 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', 'red');
% legend("Torque1 vs theta1, theta2", "Torque Limit Motor 1")
hold off;

% Surface plot for T2
subplot(1, 2, 2);
T2 = T2 ./ 3;
surf(rad2deg(Theta1), rad2deg(Theta2), T2');
title('Torque T2');
xlabel('\theta_1 (deg)');
ylabel('\theta_2 (deg)');
zlabel('T2 (Nm)');
grid on;
hold on;
% Add torque limit plane for T2
Z2_limit = t2limit * ones(size(Theta1));
surf(rad2deg(Theta1), rad2deg(Theta2), Z2_limit', 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', 'red');
% legend("Torque vs theta1, theta2", "Torque Limit Motor 1")
hold off;
