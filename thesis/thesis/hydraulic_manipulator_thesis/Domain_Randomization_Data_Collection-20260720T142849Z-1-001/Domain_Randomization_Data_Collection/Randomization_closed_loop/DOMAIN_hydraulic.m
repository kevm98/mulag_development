%% =========================================================================
% CURRICULUM DATASET GENERATION — 1000 RUNS
% MODEL: Mulag_FME_500_with_cylinder
% Structure:
%   Run 1               : Nominal (1 run)
%   Runs 2–251          : Random Hydraulics only (250 runs)
%   Runs 252–501        : Random Mechanics only (250 runs)
%   Runs 502–1000       : Fully Randomized (both) (499 runs)
%   Total: 1000 runs with Auto-Backup every 100 runs
% =========================================================================
clear; clc; close all;

% --- CONFIGURATION ---
model_name  = 'Mulag_FME_500_with_cylinder';
Target_Runs = 1000;   
T_end       = 22;     
Ts          = 0.01;   
nPts        = round(T_end / Ts) + 1;
time        = (0:nPts-1)' * Ts;

Dataset = struct();

fprintf('Loading model...\n');
load_system(model_name);

% ==================== Solver ====================
disp('Applying speed & forgiving solver settings...');
set_param(model_name, 'SimulationMode', 'accelerator');
set_param(model_name, 'Solver', 'ode23t');
set_param(model_name, 'RelTol', '1e-2');
set_param(model_name, 'AbsTol', '1e-3');
set_param(model_name, 'MaxStep', 'Inf');
set_param(model_name, 'FastRestart', 'off');
set_param(model_name, 'OutputOption', 'SpecifiedOutputTimes');
set_param(model_name, 'OutputTimes', sprintf('[0:0.05:%f]', T_end));

disp('Initializing Base Workspaces...');

%% -------------------------------------------------------------------------
% PHASE 0: ROBOT SETUP (For Joint Limits & FK)
% -------------------------------------------------------------------------
try
    robot = importrobot('Mulag_FME500.urdf');
catch
    warning('URDF not found. Creating simple 5-DOF demo robot.');
    robot = createSimpleRobot();
end
robot.DataFormat = 'column';

endEffector = 'Messerkopf';
if ~any(strcmp(robot.BodyNames, endEffector))
    endEffector = robot.BodyNames{end};
end

% The script extracts the exact limits from your URDF here and trusts them 100%
[jType, qmin, qmax, isRev, isPris] = jointMeta(robot);
JDOF = numel(qmin);

%% -------------------------------------------------------------------------
% PHASE 1: STATIC NOMINALS DEFINITION
% -------------------------------------------------------------------------
assignin('base', 'alpha_air', 0.005); assignin('base', 'rho_oil', 845);
assignin('base', 'p_displacement', 25); assignin('base', 'p_gain', 1e7);
assignin('base', 'p_velocity', 188); 
assignin('base', 'prv_PregRange', 2.759e6); assignin('base', 'prv_Amax', 0.002);
assignin('base', 'prv_Aleak', 1e-9); assignin('base', 'prv_ReCrit', 12);
assignin('base', 'prv_tau_open', 0.04); assignin('base', 'k_contact', 5e5);
assignin('base', 'b_contact', 5e3);
assignin('base', 'Iprod', [0 0 0]);

for k=1:4
    assignin('base', sprintf('cyl_head_len%d', k), 0.01);
    if k<=2, assignin('base', sprintf('A_piston%d',k), 0.2513); assignin('base', sprintf('A_annulus%d',k), 0.2513); assignin('base', sprintf('stroke%d',k), 1.85);
    else,    assignin('base', sprintf('A_piston%d',k), 0.1256); assignin('base', sprintf('A_annulus%d',k), 0.1256); assignin('base', sprintf('stroke%d',k), 2.25); end
end

for k=1:2
    assignin('base', sprintf('f_force%d',k), 2e5); assignin('base', sprintf('f_coulomb%d',k), 2e5); assignin('base', sprintf('f_velocity%d',k), 0.1);
end
assignin('base', 'f_force3', 1000); assignin('base', 'f_coulomb3', 1000); assignin('base', 'f_velocity3', 0.1);
assignin('base', 'f_force4', 2000); assignin('base', 'f_coulomb4', 2000); assignin('base', 'f_velocity4', 0.1);

% MECHANICAL NOMINALS
nom_m_link1 = 236.338; nom_com_link1 = [0.233861, -0.0608381, 0.24567]; nom_I_link1 = [40.0143, 16.8875, 45.9829]; nom_Iprod_link1 = [2.42626, 0.432578, -1.42106];
nom_m_link2 = 79.6991; nom_com_link2 = [0.00782058, -0.0531715, 0.576558]; nom_I_link2 = [5.56735, 5.41795, 0.913983]; nom_Iprod_link2 = [0.53705, 0.132287, 0.0225438];
nom_m_link3 = 106.171; nom_com_link3 = [1.56406, 0.00654027, 2.04193e-05]; nom_I_link3 = [0.631129, 80.9252, 80.9184]; nom_Iprod_link3 = [-9.55394e-05, -0.000904325, 0.512769];
nom_m_link4 = 84.1515; nom_com_link4 = [1.14397, -0.0915572, 8.74248e-05]; nom_I_link4 = [0.667361, 52.9223, 53.2899]; nom_Iprod_link4 = [-0.000361956, 0.00798074, -0.335725];
nom_m_link5 = 12.5697; nom_com_link5 = [0.0836514, -0.00277649, 0.000142508]; nom_I_link5 = [0.106121, 0.106154, 0.0967841]; nom_Iprod_link5 = [4.71246e-05, -2.66519e-06, 0.0144236];
nom_m_link6 = 248.085; nom_com_link6 = [0.280072, -0.0730586, 0.208635]; nom_I_link6 = [55.5388, 21.4578, 67.5777]; nom_Iprod_link6 = [0.678983, -0.188076, 2.93129];
nom_m_cyl = 7.12257; nom_com_cyl = [0.000849256, -3.07002e-07, 0.24945]; nom_I_cyl = [0.215291, 0.216002, 0.00723345]; nom_Iprod_cyl = [4.07841e-08, 4.24255e-05, 2.53744e-08];
nom_m_pist = 5.61889; nom_com_pist = [-1.13434e-06, -2.22099e-07, 0.261698]; nom_I_pist = [0.177141, 0.177509, 0.00171372]; nom_Iprod_pist = [-1.50897e-09, 7.7033e-07, -8.61796e-08];

%% -------------------------------------------------------------------------
% PHASE 2: OFFLINE PRE-COMPUTATION — 1000 RUNS 
% -------------------------------------------------------------------------
disp('Generating diverse Offline Parameter Table (1000 runs)...');
LHS_Matrix = lhsdesign(Target_Runs, 12); 
ValidParams = cell(Target_Runs, 1);

for i = 1:Target_Runs
    p = struct(); p.RunType = 'Nominal';
    
    is_hyd_rand  = (i >= 2 && i <= 251) || (i >= 502);
    is_mech_rand = (i >= 252 && i <= 1000);
    
    if is_hyd_rand && is_mech_rand, p.RunType = 'Fully Randomized';
    elseif is_hyd_rand, p.RunType = 'Random Hydraulics';
    elseif is_mech_rand, p.RunType = 'Random Mechanics';
    end

    if is_hyd_rand
        p.eps_air     = 0.001 + 0.079 * LHS_Matrix(i, 1);   
        p.eps_vol     = 0.005 + 0.075 * LHS_Matrix(i, 2);   
        p.delta_spool = 0.20 * (2*LHS_Matrix(i,3)-1);       
        p.w_seal      = 0.25 * LHS_Matrix(i, 4:7);          
        p.Pset_mult   = 0.90 + 0.20 * LHS_Matrix(i, 12);    
    else
        p.eps_air = 0.005; p.eps_vol = 0.02; p.delta_spool = 0; p.w_seal = zeros(1,4); p.Pset_mult = 1.0;
    end
    beta_calc_pa = 1 / ( (1 - p.eps_air)/0.5e9 + p.eps_air/(1.4 * 1e7) + p.eps_vol/0.5e9 );
    p.beta_eff   = min(max(beta_calc_pa / 1e9, 0.7), 1.8);
    p.Cd_actual  = 0.7 * (1 + p.delta_spool);
    p.prv_Pset   = 27.6e6 * p.Pset_mult;

    if is_mech_rand
        mass_var = 0.20; com_var = 0.05;
        [p.m_link1, p.com_link1, p.I_link1, p.Iprod_link1] = randomizeRigidBody(nom_m_link1, nom_com_link1, nom_I_link1, nom_Iprod_link1, mass_var, com_var);
        [p.m_link2, p.com_link2, p.I_link2, p.Iprod_link2] = randomizeRigidBody(nom_m_link2, nom_com_link2, nom_I_link2, nom_Iprod_link2, mass_var, com_var);
        [p.m_link3, p.com_link3, p.I_link3, p.Iprod_link3] = randomizeRigidBody(nom_m_link3, nom_com_link3, nom_I_link3, nom_Iprod_link3, mass_var, com_var);
        [p.m_link4, p.com_link4, p.I_link4, p.Iprod_link4] = randomizeRigidBody(nom_m_link4, nom_com_link4, nom_I_link4, nom_Iprod_link4, mass_var, com_var);
        [p.m_link5, p.com_link5, p.I_link5, p.Iprod_link5] = randomizeRigidBody(nom_m_link5, nom_com_link5, nom_I_link5, nom_Iprod_link5, mass_var, com_var);
        [p.m_link6, p.com_link6, p.I_link6, p.Iprod_link6] = randomizeRigidBody(nom_m_link6, nom_com_link6, nom_I_link6, nom_Iprod_link6, mass_var, com_var);
        for k=1:4
            [p.m_cyl{k}, p.com_cyl{k}, p.I_cyl{k}, p.Iprod_cyl{k}] = randomizeRigidBody(nom_m_cyl, nom_com_cyl, nom_I_cyl, nom_Iprod_cyl, mass_var, com_var);
            [p.m_pist{k}, p.com_pist{k}, p.I_pist{k}, p.Iprod_pist{k}] = randomizeRigidBody(nom_m_pist, nom_com_pist, nom_I_pist, nom_Iprod_pist, mass_var, com_var);
        end
        p.stif_J12 = 40000 + 80000 * LHS_Matrix(i, 8); p.stif_J34 = 40000 + 80000 * LHS_Matrix(i, 9);
        p.stif_J5  = 15000 + 45000 * LHS_Matrix(i,10); p.damp_val =  100 +  500 * LHS_Matrix(i,11);
    else
        p.m_link1 = nom_m_link1; p.com_link1 = nom_com_link1; p.I_link1 = nom_I_link1; p.Iprod_link1 = nom_Iprod_link1;
        p.m_link2 = nom_m_link2; p.com_link2 = nom_com_link2; p.I_link2 = nom_I_link2; p.Iprod_link2 = nom_Iprod_link2;
        p.m_link3 = nom_m_link3; p.com_link3 = nom_com_link3; p.I_link3 = nom_I_link3; p.Iprod_link3 = nom_Iprod_link3;
        p.m_link4 = nom_m_link4; p.com_link4 = nom_com_link4; p.I_link4 = nom_I_link4; p.Iprod_link4 = nom_Iprod_link4;
        p.m_link5 = nom_m_link5; p.com_link5 = nom_com_link5; p.I_link5 = nom_I_link5; p.Iprod_link5 = nom_Iprod_link5;
        p.m_link6 = nom_m_link6; p.com_link6 = nom_com_link6; p.I_link6 = nom_I_link6; p.Iprod_link6 = nom_Iprod_link6;
        for k=1:4
            p.m_cyl{k} = nom_m_cyl; p.com_cyl{k} = nom_com_cyl; p.I_cyl{k} = nom_I_cyl; p.Iprod_cyl{k} = nom_Iprod_cyl;
            p.m_pist{k} = nom_m_pist; p.com_pist{k} = nom_com_pist; p.I_pist{k} = nom_I_pist; p.Iprod_pist{k} = nom_Iprod_pist;
        end
        p.stif_J12 = 60000; p.stif_J34 = 60000; p.stif_J5 = 25000; p.damp_val = 250;
    end
    ValidParams{i} = p;
end

%% -------------------------------------------------------------------------
% PHASE 3: EXECUTION LOOP
% -------------------------------------------------------------------------
for i = 1:Target_Runs
    p = ValidParams{i};
    fprintf('\n--- Generating Entry %d / %d [%s] ---\n', i, Target_Runs, p.RunType);
    
    % ====================================================================
    % IMPROVED TRAJECTORY GENERATION FOR RL TRAINING
    % Changes from original:
    %   - Increased waypoints: 12-20 instead of 4-8
    %   - Removed dwell times (no static holding)
    %   - Added speed diversity (slow/medium/fast segments)
    %   - Added exploration perturbations (30% of runs)
    % ====================================================================
    
    % STEP 1: Generate more waypoints for better coverage
    num_poses = randi([12, 20]);  % CHANGED FROM [4, 8]
    lhs_points = lhsdesign(num_poses, JDOF);
    
    % STEP 2: Add speed diversity
    speed_types = {'slow', 'medium', 'fast', 'variable'};
    segment_speeds = speed_types(randi(4, [num_poses, 1]));
    
    % STEP 3: Build waypoints with variable timing (NO DWELL)
    q_waypoints = zeros(JDOF, num_poses);
    t_waypoints = zeros(1, num_poses);
    
    current_time = 0;
    for pose_idx = 1:num_poses
        % Sample target configuration from LHS
        target_q = zeros(JDOF, 1);
        for j = 1:JDOF
            % Pull direct from URDF
            min_val = qmin(j); 
            max_val = qmax(j);
            
            % Guard strictly for completely continuous/infinite joints
            if isinf(min_val), min_val = -pi; end
            if isinf(max_val), max_val = pi; end
            
            target_q(j) = min_val + (max_val - min_val) * lhs_points(pose_idx, j);
        end
        
        q_waypoints(:, pose_idx) = target_q;
        
        % Variable speed timing (REMOVED DWELL - just move to next pose)
        switch segment_speeds{pose_idx}
            case 'slow'
                segment_time = 1.5 + 1.0 * rand();  % 1.5-2.5 sec
            case 'medium'
                segment_time = 0.8 + 0.7 * rand();  % 0.8-1.5 sec
            case 'fast'
                segment_time = 0.3 + 0.4 * rand();  % 0.3-0.7 sec
            case 'variable'
                segment_time = 0.5 + 2.0 * rand();  % 0.5-2.5 sec (wide range)
        end
        
        current_time = current_time + segment_time;
        t_waypoints(pose_idx) = current_time;
    end
    
    % Normalize time to exactly hit T_end (22 seconds)
    t_waypoints = (t_waypoints / t_waypoints(end)) * T_end;
    
    % STEP 4: Interpolate smooth trajectory
    q_ref = zeros(JDOF, nPts);
    for j = 1:JDOF
        q_ref(j,:) = pchip(t_waypoints, q_waypoints(j,:), time');
    end
    
    % STEP 5: Add exploration perturbations (30% of runs)
    if rand() < 0.3
        for j = 1:JDOF
            % Low-frequency sinusoid to explore local neighborhoods
            freq = 0.1 + 0.3 * rand();  % 0.1-0.4 Hz
            amplitude = 0.05 * (qmax(j) - qmin(j));  % 5% of joint range
            q_ref(j,:) = q_ref(j,:) + amplitude * sin(2*pi*freq*time');
            
            % Clamp to limits
            q_ref(j,:) = max(min(q_ref(j,:), qmax(j)), qmin(j));
        end
    end
    
    % STEP 6: Calculate Derivatives
    dq_ref = zeros(JDOF, nPts);
    ddq_ref = zeros(JDOF, nPts);
    for j = 1:JDOF
        dq_ref(j,:)  = gradient(q_ref(j,:), Ts);
        ddq_ref(j,:) = gradient(dq_ref(j,:), Ts);
    end
    
    % STEP 7: Forward Kinematics
    ee_actual = zeros(3, nPts);
    for k = 1:nPts
        T_act = getTransform(robot, q_ref(:,k), endEffector);
        ee_actual(:,k) = T_act(1:3,4);
    end
    
    % STEP 8: Compute quality metrics for logging
    % Joint space coverage
    coverage = zeros(JDOF, 1);
    for j = 1:JDOF
        q_range = qmax(j) - qmin(j);
        if isinf(q_range), q_range = 2*pi; end
        q_explored = max(q_ref(j,:)) - min(q_ref(j,:));
        coverage(j) = q_explored / q_range;
    end
    mean_coverage = mean(coverage);
    
    % Velocity diversity
    mean_vel_diversity = mean(std(abs(dq_ref), [], 2));
    
    % Static time percentage
    max_vels = max(abs(dq_ref), [], 1);
    static_count = sum(max_vels < 0.01 * max(max_vels));
    static_percentage = 100 * static_count / nPts;
    
    % ====================================================================
    % INJECT TRAJECTORY DIRECTLY TO SIMULINK FROM-WORKSPACE BLOCKS
    % ====================================================================
    assignin('base', 'pos_ref_data', timeseries(q_ref', time));
    assignin('base', 'vel_ref_data', timeseries(dq_ref', time));
    assignin('base', 'acc_ref_data', timeseries(ddq_ref', time));
    assignin('base', 'ee_ref_data',  timeseries(ee_actual', time));
    
    % --- INJECT HYDRAULICS ---
    assignin('base', 'beta_oil', p.beta_eff); assignin('base', 'prv_Cd', p.Cd_actual); assignin('base', 'prv_Pset', p.prv_Pset); 
    for k=1:4
        assignin('base', sprintf('cyl_clearance%d', k), 1e-4 * (1 + p.w_seal(k)));
        assignin('base', sprintf('A_leak_valve%d', k), 1e-12 * (1 + p.w_seal(k))^3);
        assignin('base', sprintf('tau_valve%d', k), 0.008 * (1 + 2.5 * p.w_seal(k)));
        Vd = min(max(5e-4 * (1 + p.eps_air + p.eps_vol), 5e-4), 1e-3);
        assignin('base', sprintf('V_dead_A%d', k), Vd); assignin('base', sprintf('V_dead_B%d', k), Vd);
    end
    
    % --- INJECT MECHANICS ---
    assignin('base', 'm_link1', p.m_link1); assignin('base', 'com_link1', p.com_link1); assignin('base', 'I_link1', p.I_link1); assignin('base', 'Iprod_link1', p.Iprod_link1);
    assignin('base', 'm_link2', p.m_link2); assignin('base', 'com_link2', p.com_link2); assignin('base', 'I_link2', p.I_link2); assignin('base', 'Iprod_link2', p.Iprod_link2);
    assignin('base', 'm_link3', p.m_link3); assignin('base', 'com_link3', p.com_link3); assignin('base', 'I_link3', p.I_link3); assignin('base', 'Iprod_link3', p.Iprod_link3);
    assignin('base', 'm_link4', p.m_link4); assignin('base', 'com_link4', p.com_link4); assignin('base', 'I_link4', p.I_link4); assignin('base', 'Iprod_link4', p.Iprod_link4);
    assignin('base', 'm_link5', p.m_link5); assignin('base', 'com_link5', p.com_link5); assignin('base', 'I_link5', p.I_link5); assignin('base', 'Iprod_link5', p.Iprod_link5);
    assignin('base', 'm_link6', p.m_link6); assignin('base', 'com_link6', p.com_link6); assignin('base', 'I_link6', p.I_link6); assignin('base', 'Iprod_link6', p.Iprod_link6);
    
    for k=1:4
        assignin('base', sprintf('m_cyl%d',k), p.m_cyl{k}); assignin('base', sprintf('com_cyl%d',k), p.com_cyl{k}); assignin('base', sprintf('I_cyl%d',k), p.I_cyl{k}); assignin('base', sprintf('Iprod_cyl%d',k), p.Iprod_cyl{k});
        assignin('base', sprintf('m_pist%d',k), p.m_pist{k}); assignin('base', sprintf('com_pist%d',k), p.com_pist{k}); assignin('base', sprintf('I_pist%d',k), p.I_pist{k}); assignin('base', sprintf('Iprod_pist%d',k), p.Iprod_pist{k});
        assignin('base', 'Iprod', p.Iprod_pist{k});
    end
    
    % --- INJECT JOINTS ---
    assignin('base', 'J1_stif', p.stif_J12); assignin('base', 'J1_damp', p.damp_val);
    assignin('base', 'J2_stif', p.stif_J12); assignin('base', 'J2_damp', p.damp_val);
    assignin('base', 'J3_stif', p.stif_J34); assignin('base', 'J3_damp', p.damp_val);
    assignin('base', 'J4_stif', p.stif_J34); assignin('base', 'J4_damp', p.damp_val);
    assignin('base', 'J5_stif', p.stif_J5);  assignin('base', 'J5_damp', p.damp_val);
    
    % --- SIMULATE ---
    try
        set_param(model_name, 'StopTime', num2str(T_end));
        SimIn = Simulink.SimulationInput(model_name);
        simOut = sim(SimIn);
        
        logs = simOut.logsout;
        ts_meas = safe_get(logs, 'ee_meas');
        if isempty(ts_meas), ts_meas = safe_get(logs, 'cmd1'); end
        
        % Package Run Data for Dataset
        RunData = struct(); RunData.Params = p; RunData.Time = ts_meas.Time;
        RunData.pos_ref = q_ref; 
        RunData.vel_ref = dq_ref;
        
        for k=1:5
            RunData.ValveCmd(:,k) = interp_safe(safe_get(logs, sprintf('cmd%d', k)), RunData.Time);
        end
        for k=1:4
            ts_PA = safe_get(logs, sprintf('PA%d', k)); ts_PB = safe_get(logs, sprintf('PB%d', k));
            RunData.dP(:,k) = interp_safe(ts_PA, RunData.Time) - interp_safe(ts_PB, RunData.Time);
            RunData.Force(:,k) = interp_safe(safe_get(logs, sprintf('F%d', k)), RunData.Time);
            RunData.Extension(:,k)= interp_safe(safe_get(logs, sprintf('Ext%d', k)), RunData.Time);
        end
        for k=1:5
            RunData.JointPos(:,k) = interp_safe(safe_get(logs, sprintf('J%d_pos', k)), RunData.Time);
            RunData.JointVel(:,k) = interp_safe(safe_get(logs, sprintf('J%d_vel', k)), RunData.Time);
        end
        
        [J_total, Metrics] = Evaluate_Performance_Stability(logs);
        
        Dataset(i).Data = RunData; Dataset(i).Score = J_total; Dataset(i).Metrics = Metrics;
        
        % Store trajectory quality metrics
        Dataset(i).TrajMetrics.num_poses = num_poses;
        Dataset(i).TrajMetrics.mean_coverage = mean_coverage;
        Dataset(i).TrajMetrics.mean_vel_diversity = mean_vel_diversity;
        Dataset(i).TrajMetrics.static_percentage = static_percentage;
        
        fprintf(' [+] SUCCESS %d/%d (Poses: %d, Coverage: %.1f%%, Static: %.1f%%) | Score: %.4f\n', ...
            i, Target_Runs, num_poses, mean_coverage*100, static_percentage, J_total);
        
        % Auto-Save Backup
        if mod(i, 100) == 0
            backup_name = sprintf('Hydraulic_Dataset_Backup_Run_%d.mat', i);
            save(backup_name, 'Dataset', '-v7.3');
            fprintf(' ---> Auto-saved backup: %s\n', backup_name);
        end
        
    catch ME
        fprintf('\n [!] CRASH IN RUN %d:\n', i); disp(ME.getReport('extended', 'hyperlinks', 'off'));
        fprintf('\n[STOPPING SCRIPT] Fix the error printed above before continuing.\n'); break;
    end
end

if exist('Dataset', 'var') && length(Dataset) == Target_Runs
    save('Hydraulic_Mechanic_Dataset_Curriculum_1000.mat', 'Dataset', '-v7.3');
    disp('Dataset collection complete! 1000-run curriculum successfully executed.');
    
    % Print summary statistics
    fprintf('\n========== DATASET SUMMARY ==========\n');
    all_coverage = arrayfun(@(x) x.TrajMetrics.mean_coverage, Dataset);
    all_static = arrayfun(@(x) x.TrajMetrics.static_percentage, Dataset);
    all_poses = arrayfun(@(x) x.TrajMetrics.num_poses, Dataset);
    
    fprintf('Trajectory Quality Across 1000 Runs:\n');
    fprintf('  Mean Joint Coverage:   %.1f%% (std: %.1f%%)\n', mean(all_coverage)*100, std(all_coverage)*100);
    fprintf('  Mean Static Time:      %.1f%% (std: %.1f%%)\n', mean(all_static), std(all_static));
    fprintf('  Mean Poses per Run:    %.1f (range: %d-%d)\n', mean(all_poses), min(all_poses), max(all_poses));
    fprintf('======================================\n');
end

%% --- HELPERS ---
function [m_new, com_new, I_moments, I_products] = randomizeRigidBody(m_nom, com_nom, I_nom_moments, I_nom_products, mass_var, com_var)
    if nargin < 5, mass_var = 0.35; end; if nargin < 6, com_var = 0.05; end
    s = 1 + mass_var * (2*rand() - 1); m_new = m_nom * s;
    delta = randn(3,1); delta = delta/norm(delta) * com_var * rand(); com_new = com_nom(:) + delta;
    I_moments = I_nom_moments * (m_new/m_nom); I_products = I_nom_products * (m_new/m_nom); com_new = com_new';
end

function ts = safe_get(logs, searchString)
    ts = []; el = logs.get(searchString);
    if ~isempty(el), ts = el.Values; ts.Data = squeeze(ts.Data); return; end
    if contains(searchString, '/'), parts = strsplit(searchString, '/'); targetName = parts{end}; else, targetName = searchString; end
    allNames = logs.getElementNames;
    for i = 1:numel(allNames)
        if endsWith(allNames{i}, targetName) || strcmp(allNames{i}, targetName)
            ts = logs.get(i).Values; ts.Data = squeeze(ts.Data); return; 
        end
    end
end

function data_out = interp_safe(ts, t_target)
    if isempty(ts) || isempty(ts.Data), data_out = zeros(size(t_target)); return; end
    x = ts.Time; v = squeeze(ts.Data);
    if size(v,1) ~= length(x) && size(v,2) == length(x), v = v'; end
    try data_out = interp1(x, v, t_target, 'linear', 'extrap'); catch, data_out = zeros(size(t_target)); end
end

function [J_total, res] = Evaluate_Performance_Stability(logs)
    W.track = 0.60; W.stab = 0.40;
    ts_ref = safe_get(logs, 'ee_ref'); ts_meas = safe_get(logs, 'ee_meas');
    if isempty(ts_ref) || isempty(ts_meas), J_total=NaN; res.J_track=NaN; res.J_oscill=NaN; return; end
    x_sim = interp_safe(ts_meas, ts_ref.Time); err = ts_ref.Data - x_sim;
    if size(err,2)>1, J_track = sqrt(mean(sum(err.^2,2))); else, J_track = sqrt(mean(err.^2)); end
    
    J_oscill = 0;
    for k=1:4, ts_PA = safe_get(logs, sprintf('PA%d',k)); if ~isempty(ts_PA), J_oscill = J_oscill + std(diff(ts_PA.Data)); end; end
    J_oscill = J_oscill / 1e5; J_total = (W.track * J_track) + (W.stab * J_oscill); res.J_track = J_track; res.J_oscill = J_oscill;
end

function [jType,qmin,qmax,isRev,isPris] = jointMeta(robot)
    fmt0 = robot.DataFormat; robot.DataFormat = 'struct'; cfgS = homeConfiguration(robot); robot.DataFormat = fmt0;
    n = numel(cfgS); jType = strings(n,1); qmin = -inf(n,1); qmax = inf(n,1);
    bodies = robot.Bodies; jointMap = containers.Map('KeyType','char','ValueType','any');
    for i = 1:numel(bodies), J = bodies{i}.Joint; if ~strcmp(J.Type,'fixed'), jointMap(J.Name) = J; end; end
    for i = 1:n
        jname = cfgS(i).JointName;
        if isKey(jointMap, jname)
            J = jointMap(jname); jType(i) = string(J.Type); lim = J.PositionLimits;
            if ~isempty(lim), qmin(i) = lim(1); qmax(i) = lim(2); if qmin(i) == qmax(i), qmin(i) = -inf; qmax(i) = inf; end; end
        else, jType(i) = "revolute"; qmin(i) = -pi; qmax(i) = pi; end
    end
    isRev = (jType == "revolute"); isPris = (jType == "prismatic");
end

function robot = createSimpleRobot()
    L1=0.5; L2=0.5; L3=0.4; L4=0.3; L5=0.25; robot = rigidBodyTree('MaxNumBodies',6,'DataFormat','column');
    b1 = rigidBody('b1'); j1 = rigidBodyJoint('j1','revolute'); setFixedTransform(j1,trvec2tform([0 0 0])); b1.Joint = j1; addBody(robot,b1,'base');
    b2 = rigidBody('b2'); j2 = rigidBodyJoint('j2','revolute'); setFixedTransform(j2,trvec2tform([0 0 L1])); b2.Joint = j2; addBody(robot,b2,'b1');
    b3 = rigidBody('b3'); j3 = rigidBodyJoint('j3','revolute'); setFixedTransform(j3,trvec2tform([L2 0 0])); b3.Joint = j3; addBody(robot,b3,'b2');
    b4 = rigidBody('b4'); j4 = rigidBodyJoint('j4','revolute'); setFixedTransform(j4,trvec2tform([L3 0 0])); b4.Joint = j4; addBody(robot,b4,'b3');
    b5 = rigidBody('b5'); j5 = rigidBodyJoint('j5','revolute'); setFixedTransform(j5,trvec2tform([L4 0 0])); b5.Joint = j5; addBody(robot,b5,'b4');
    bee = rigidBody('Messerkopf'); je = rigidBodyJoint('je','fixed'); setFixedTransform(je,trvec2tform([L5 0 0])); bee.Joint = je; addBody(robot,bee,'b5');
end