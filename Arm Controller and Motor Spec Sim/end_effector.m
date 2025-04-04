m_payload = 4 ; % kg 
a = 0.25; % m/s ^ 2
g = 9.81 ; % m/s^2
u = 0.75; % coefficient of friction
fos = 1.5;
F = fos * m_payload * (g + a) / u;
arm_distance = 0.015; % m
torque = F * arm_distance % Nm

% Motor torque arm : 