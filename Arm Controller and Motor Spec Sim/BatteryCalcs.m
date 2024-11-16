%% Read data
torque = table2array(readtable("torques.csv"));
velocity = table2array(readtable("vels.csv"));
time = table2array(readtable("times.csv"));

%% Calculation

% torque and velocity trajectories already determined based on tasks

voltage = 36; % V
mech_power = torque .* velocity; % W (N/m * m/s) each individual motor
n_motor = 0.9; % Efficiency of motor
n_controller = 0.9; % Efficiency of controller
fos = 1.5;

% Calculate electrical power for each motor
electrical_power = mech_power ./ (n_motor * n_controller); % W
total_power = sum(electrical_power, 2); % W
electrical_energy = trapz(time, total_power) / 3600; % Wh

% Assume we cycle the same trajectories
n_cycle = 100;
electrical_energy_cycled = electrical_energy * n_cycle;
operating_time_minutes = n_cycle * time(end) / 60

% Calculate capacity with FOS
required_capacity_Ah = fos * electrical_energy_cycled / voltage

% Calculate currents
kt = [1.22, 2.36, 1.22, 3.66]; % torque constant for each motor
current = torque./kt;
current_sum = sum(current, 2);
peak_current = max(current_sum)
average_current = mean(current_sum)

% Calculate C-rate
c_rate = peak_current / required_capacity_Ah;

%% Chose battery specs : https://www.amazon.com/HAILONG-Battery-Electric-Scooter-Lithium/dp/B09JNVQ19B?crid=2MP1YAGKX2ZQ6&dib=eyJ2IjoiMSJ9.Xn_5_pgsp-zMuGpdQV5QIhVOI2vbpqXuZVra1YEV3Sy9V7AAIIus9k8YdSsRlou9MlPSluMjYlm-kN6M6u_nm2aGTEZHd9ATpxo0iU8QDTKAVZTL3X4-YCXMH08XK6j_ldyaZkCAtQ8ZtlPWd_SPnP8it616ZnccncqdKOB5J9FnWPXvrFf5wozCRVS1zNmiy8Gx3nOOhjvB1UbgR-270cDX3XTl5h7DxAheigjeCBdIneG-kVGnRk_EdkwEZmCX6gC4oCWygiYczFUWajSQfU5ixWXIQ5nKEYGHMRnNIBI.Q_2ixrEvflwuCWe3mG6liVvQlAT0wqp7alYeNzq2VAA&dib_tag=se&keywords=36+v+battery&qid=1731775760&sprefix=36+v+battery%2Caps%2C204&sr=8-5
nominal_voltage = 36 ; %v
rated_capacity = 8; % Ah
discharge_range = 0.8; % Adding a factor to account for the fact that we should not discharge the battery 100%
continuous_current = 20; % A (conservative, since our peak current draw is 21 A)
lower_bound_operating_time = discharge_range * rated_capacity / continuous_current * 60 % minutes
upper_bound_operating_time = discharge_range * rated_capacity / average_current * 60 % minutes

% Config : 10s 3p
% Dimensions : 198 * 70 *70 mm
% weight = 1.7 kg
% Continuous current = 20 Amp , Peak current = 60 Amp
% All 4 motors we be in parallel to the battery