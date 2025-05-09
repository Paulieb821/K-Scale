%% Belt Forces

d = 20.62e-3 % m (driving pulley diameter)
D = 57.302e-3 % m (driven pulley diameter)
C = 0.308 % m (inter pulley distance)
T = 8 % Motor torque Nm
gamma = 1500 % belt material density (neoprene upperbound) kg/m^3
b = 9e-3 % belt width : m
t = 2.46e-3 % belt thickness : m
n = 7 %  rotational speed of driving pulley rad/s 
w = gamma * b * t % belt mass per unit length kg / m
g = 9.81 % gravity 
V = pi * d * n % belt velocity : m/s
Fc = w / g * (V / 60) ^ 2 % Centrifigual force Newtons eq e
f = 0.5 % Coefficient of friction belt - rough guess based on shigley
phi_small = pi - 2 * asin((D - d) / (2 * C)) % wrap angle for small pulley eq 17-1
phi_big = pi + 2 * asin((D - d) / (2 * C)) % wrap angle for big pulley 
Fi = T * exp(f * phi_small) + 1 / (d * exp(f * phi_small) - 1) % initial tension N eq i
F2 = Fi + Fc - T/d % slack side pulley tension N 
F1 = Fi + Fc + T/d % tight side pulley tension N
Fr = sqrt(F1^2 + F2^2 - 2 * abs(F1 * F2)* cos(phi_small)) % Resultant force on shaft N

fprintf('Tight side tension (Ft): %.2f N\n', F1);
fprintf('Slack side tension (Fs): %.2f N\n', F2);
fprintf('Resultant belt force (Fb): %.2f N\n', Fr);
% belt dimensions

min_belt_length = (sqrt(4 * C^2 - (D - d)^2) + 0.5 * (D * phi_big + d * phi_big)) * 1000; % mm
belt_length = 735;
center_center_distance = 0.5 * sqrt((belt_length/1000 - 0.5*phi_big*(D+d))^2 + (D-d)^2)


%% Key failure check 
%% Key and Shaft Failure Torque Estimation
% This script estimates the torque at which:
%   1) A rectangular key will fail in shear or bearing.
%   2) The shaft will fail in torsion (simplified).

clear; clc;

%% Key shear stress and bearing stress calcs

% Key geometry [mm]
key_width  = 2;
key_height = 2;
key_length = 22;

% Key properties
Sy_key = 300e6;    % [Pa] yield strength of the key
tau_allow_key = 0.5 * Sy_key;    % [Pa] allowable shear stress in the key
sigma_allow_key = 0.9 * Sy_key;  % [Pa] allowable bearing (compressive) stress

% Shaft geometry
shaft_diameter = 5/16 * 25.4;
shaft_radius   = shaft_diameter / 2;

%Shaft properties : Assume Aluminum 6061
Sy_shaft = 276.e6; % Pa : yield strength shaft
tau_allow_shaft = 0.6 * Sy_shaft;

N = 1.5; % FOS

% Shear stress on key
A_shear_key = key_width * key_length * 1e-6;  % [m^2]
T_key_shear = tau_allow_key * A_shear_key * (shaft_radius * 1e-3); % [N·m]

% Bearing stress
A_bearing_key = key_height * key_length * 1e-6;   % [m^2]
T_key_bearing = sigma_allow_key * A_bearing_key * (shaft_radius * 1e-3); % [N·m]

% The actual key torque rating is limited by whichever is smaller:
T_key_fail = min(T_key_shear, T_key_bearing);

% Shaft yield stress
Kt = 1;
d_m = (shaft_diameter) * 1e-3;    % convert mm to m and apply the keyway offset
T_shaft_yield = Kt * tau_allow_shaft * (pi * d_m^3 / 16); % [N·m]

% Set Screw Compensation
scaling_factor = 1.8; % We have 2 set screws , this is a conservative assumption on how holding torque will scale (needs more investigation)
set_screw_power = scaling_factor * 60 * 0.112984829 ; % (lb-in converted to Nm) 8-32 set screw, used Unbrako chart page 22


% Apply factor of safety:
T_key_fail_N = (T_key_fail + set_screw_power) / N;
T_shaft_yield_N = (T_shaft_yield + set_screw_power) / N;

% pay-load capacity 
load_end_effector = 4 * 9.81 ; % 4kg
weight_forearm = 2 * 9.81 ; % 2kg
length_forearm = 0.25 ; % m
total_moment = load_end_effector * length_forearm + weight_forearm * length_forearm / 2;

% 4. DISPLAY RESULTS
fprintf('Set screw power reduces total amount of torque on the keys by %.2f Nm\n', set_screw_power)
fprintf('\n=== KEY FAILURE MODES ===\n');
fprintf('  Shear-limited torque   = %.2f N·m\n', T_key_shear);
fprintf('  Bearing-limited torque = %.2f N·m\n', T_key_bearing);
fprintf('  Key limiting torque    = %.2f N·m  (with FoS = %.2f)\n', ...
    T_key_fail_N, N);
fprintf('\n=== SHAFT FAILURE IN TORSION ===\n');
fprintf('  Shaft torque at yield  = %.2f N·m  (with FoS = %.2f)\n', ...
    T_shaft_yield_N, N);

