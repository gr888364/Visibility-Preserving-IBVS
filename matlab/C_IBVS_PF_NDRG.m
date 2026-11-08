clc; clear; close all;
clear classes;

% Comparative simulation of Section 4.1 (OG-VFVR-IBVS, MPC-IBVS, APF-IBVS, P-IBVS),
% each run without and with the ND displacement-increment shaping G_t.
project_dir = fileparts(mfilename('fullpath'));
cd(project_dir);
addpath(project_dir, '-begin');
% Optional local copies of the Corke toolboxes (see README); otherwise use the MATLAB path.
tb_dirs = {'spatial-math', 'vision', 'common'};
for i_tb = 1:numel(tb_dirs)
    if exist(fullfile(project_dir, tb_dirs{i_tb}), 'dir')
        addpath(fullfile(project_dir, tb_dirs{i_tb}), '-begin');
    end
end
if exist(fullfile(project_dir, 'Camera'), 'dir')
    addpath(genpath(fullfile(project_dir, 'Camera')), '-begin');
end
which_ibvs = which('IBVS');
expected_ibvs = fullfile(project_dir, 'IBVS.m');
if ~strcmpi(which_ibvs, expected_ibvs)
    error('C_IBVS_PF_NDRG:WrongIBVSPath', ...
        'MATLAB is using the wrong IBVS class:\n  %s\nExpected:\n  %s\nRun this script from the matlab/ folder of this repository or clear the MATLAB path cache.', ...
        which_ibvs, expected_ibvs);
end

out_dir = fullfile(pwd, 'compare_outputs', 'unified_four_method_compare');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

scene = make_experiment_scene();
method_specs = make_method_specs(scene);
variant_names = {'noND', 'withND'};
variant_labels = {'Baseline', 'ND method'};
dt_seq = make_dt_sequence(scene);

fprintf('\n=== Unified IBVS Comparison ===\n');
fprintf('Scene: near-boundary setup, fps=%d, depth=%.2f, max_iter=%d, adaptive gain_type=%d, random_dt=%s\n', ...
    scene.fps, scene.depth, scene.max_iter, scene.gain_type, ternary(scene.use_random_dt, 'on', 'off'));
fprintf('Sampling: dt in [%d, %d] ms, mean %.2f ms\n', ...
    min(dt_seq), max(dt_seq), mean(dt_seq));
fprintf('Initial pose: SE3(%.2f, %.2f, %.2f) * SE3.Rz(5*pi/4)\n', ...
    scene.initial_t(1), scene.initial_t(2), scene.initial_t(3));
fprintf('Results directory:\n%s\n\n', out_dir);

results = repmat(empty_result(), numel(method_specs), numel(variant_names));
for v = 1:numel(variant_names)
    use_nd = strcmpi(variant_names{v}, 'withND');
    for i = 1:numel(method_specs)
        spec = method_specs(i);
        fprintf('Running %-24s [%s] ... ', spec.name, variant_labels{v});
        tic;
        results(i,v) = run_one_case(spec, scene, dt_seq, use_nd);
        elapsed = toc;
        fprintf('status=%d, steps=%d, final method-feature error=%.3f px, %.2f s\n', ...
            results(i,v).status, results(i,v).metrics.steps, ...
            results(i,v).metrics.final_error, elapsed);
    end
end

for v = 1:numel(variant_names)
    suffix = variant_names{v};
    label = variant_labels{v};
    plot_3d_trajectories(results(:,v), scene, out_dir, suffix, label);
    plot_image_feature_trajectories(results(:,v), scene, out_dir, suffix, label);
    plot_jacobian_condition(results(:,v), out_dir, suffix, label);
    plot_feature_error(results(:,v), scene, out_dir, suffix, label);
    plot_control_feature_error(results(:,v), scene, out_dir, suffix, label);
    plot_executed_displacement(results(:,v), out_dir, suffix, label);
end

plot_executed_displacement_nd_compare(results, out_dir);
plot_no_nd_end_effector_velocity(results(:,1), out_dir);
plot_nd_pair_feature_error(results, scene, out_dir);
plot_nd_pair_increment_norm(results, out_dir);
plot_cheng_virtual_fov_diagnostics(results, scene, out_dir);

metrics_table = build_metrics_table(results, variant_names);
writetable(metrics_table, fullfile(out_dir, 'unified_four_method_metrics.csv'));
write_failure_summary(results, out_dir);
save(fullfile(out_dir, 'unified_four_method_histories.mat'), ...
    'results', 'scene', 'method_specs', 'variant_names', 'variant_labels', 'dt_seq');

fprintf('\nSaved unified figures, metrics, and histories to:\n%s\n', out_dir);


function scene = make_experiment_scene()
    scene.fps = 33;
    scene.dt_ms = round(1000 / scene.fps);
    scene.max_iter = 3000;
    scene.depth = 1;
    scene.lambda = 0.10;
    scene.eterm = 0.5;
    scene.gain_type = 1;
    scene.use_random_dt = true;
    scene.random_seed = 42;
    scene.initial_t = [-2.1 0 -3.0];
    scene.plot3d_xlim = [-2.5 0.2];
    scene.plot3d_ylim = [-1.2 0.8];
    scene.plot3d_zlim = [-4.2 -0.8];

    cam = CentralCamera('default');
    scene.npix = cam.npix;
    % Match the actual controlled target used by the original good-ND runner.
    % That script defines a z=3 grid for a preview projection, but does not
    % pass it into IBVS; the controller therefore uses the default z=0 grid.
    scene.P = mkgrid(2, 0.5);
    scene.pd = bsxfun(@plus, 200 * [-1 -1 1 1; -1 1 1 -1], cam.pp');
    scene.T_C0 = SE3(scene.initial_t(1), scene.initial_t(2), scene.initial_t(3)) * SE3.Rz(5*pi/4);
    scene.p0 = cam.project(scene.P, 'pose', scene.T_C0);
end


function specs = make_method_specs(scene)
    specs = repmat(struct( ...
        'key', '', 'name', '', 'short_name', '', 'lambda', scene.lambda, ...
        'gain_type', scene.gain_type, ...
        'lambda_adaptive_enabled', false, 'lambda_min', scene.lambda, ...
        'lambda_max', scene.lambda, 'lambda_exp_rate', 12.0, ...
        'lambda_exp_center', 0.80, 'lambda_ramp_steps', 1, ...
        'legacy_fov_enabled', false, 'legacy_fov_margin', 100, ...
        'pf_enabled', false, 'paper_apf_enabled', false, 'mpc_enabled', false, ...
        'pf_attract_gain', 2.0, 'pf_repulse_gain', 10000.0, 'pf_safety_margin', 50, ...
        'paper_theta_gain', 0.10, 'paper_sigma_gain', 0.008, ...
        'paper_repulse_gain', 5e5, 'paper_rho0', 100, 'paper_mu', 0.8, ...
        'paper_tz_min', -0.6, 'paper_tz_max', 0.25, 'paper_wz_max', 0.10, ...
        'paper_theta_pair', [1 3], ...
        'mpc_horizon', 1, 'mpc_q_weight', eye(numel(scene.pd)) * 1, ...
        'mpc_r_weight', eye(6) * 50, 'mpc_slack_weight', 1e5, ...
        'mpc_v_max', 0.50, 'mpc_w_max', 0.50, 'mpc_fov_margin', 20, ...
        'mpc_solve_repeats', 1), 1, 4);

    specs(1).key = 'classical';
    specs(1).name = 'OG-VFVR-IBVS';
    specs(1).short_name = 'OG-VFVR';
    specs(1).legacy_fov_enabled = true;

    specs(2).key = 'mpc';
    specs(2).name = 'MPC-IBVS';
    specs(2).short_name = 'MPC';
    specs(2).mpc_enabled = true;
    specs(2).mpc_horizon = 2;
    specs(2).mpc_r_weight = eye(6) * 200;
    specs(2).mpc_v_max = 0.25;
    specs(2).mpc_w_max = 0.25;
    specs(2).mpc_solve_repeats = 20;

    specs(3).key = 'corke';
    specs(3).name = 'APF-IBVS';
    specs(3).short_name = 'APF';
    specs(3).lambda = 0.4;
    specs(3).paper_apf_enabled = true;
    specs(3).paper_theta_gain = 0.25;
    specs(3).paper_sigma_gain = 0.020;
    specs(3).paper_repulse_gain = 0.01;
    specs(3).paper_rho0 = 50;
    specs(3).paper_mu = 0.8;
    specs(3).paper_tz_min = -0.10;
    specs(3).paper_tz_max = 1.50;
    specs(3).paper_wz_max = 0.25;
    specs(3).paper_theta_pair = [1 3];

    specs(4).key = 'pf_ndrg';
    specs(4).name = 'P-IBVS (proposed)';
    specs(4).short_name = 'P-IBVS';
    specs(4).lambda = 0.08;
    specs(4).lambda_adaptive_enabled = true;
    specs(4).lambda_min = 0.08;
    specs(4).lambda_max = 2.55;
    specs(4).lambda_exp_rate = 10.0;
    specs(4).lambda_exp_center = 0.74;
    specs(4).lambda_ramp_steps = 300;
    specs(4).pf_enabled = true;
end


function result = empty_result()
    result = struct( ...
        'key', '', 'name', '', 'short_name', '', 'variant', '', ...
        'use_nd', false, 'status', NaN, 'history', [], ...
        'data', [], 'metrics', []);
end


function result = run_one_case(spec, scene, dt_seq, use_nd)
    cam = CentralCamera('default');
    ibvs = IBVS(cam, 'pose0', scene.T_C0, 'pstar', scene.pd, ...
        'fps', scene.fps, 'depth', scene.depth, ...
        'lambda', spec.lambda, 'niter', scene.max_iter, 'eterm', scene.eterm);

    ibvs.gain_type = spec.gain_type;
    ibvs.lambda_adaptive_enabled = spec.lambda_adaptive_enabled;
    ibvs.lambda_min = spec.lambda_min;
    ibvs.lambda_max = spec.lambda_max;
    ibvs.lambda_exp_rate = spec.lambda_exp_rate;
    ibvs.lambda_exp_center = spec.lambda_exp_center;
    ibvs.lambda_ramp_steps = spec.lambda_ramp_steps;
    ibvs.legacy_fov_enabled = spec.legacy_fov_enabled;
    ibvs.legacy_fov_margin = spec.legacy_fov_margin;
    ibvs.nd_enabled = use_nd;
    ibvs.nd_filter_only = false;
    ibvs.pf_enabled = spec.pf_enabled;
    ibvs.paper_apf_enabled = spec.paper_apf_enabled;
    ibvs.mpc_enabled = spec.mpc_enabled;

    ibvs.pf_attract_gain = spec.pf_attract_gain;
    ibvs.pf_repulse_gain = spec.pf_repulse_gain;
    ibvs.pf_safety_margin = spec.pf_safety_margin;

    ibvs.paper_theta_gain = spec.paper_theta_gain;
    ibvs.paper_sigma_gain = spec.paper_sigma_gain;
    ibvs.paper_repulse_gain = spec.paper_repulse_gain;
    ibvs.paper_rho0 = spec.paper_rho0;
    ibvs.paper_mu = spec.paper_mu;
    ibvs.paper_tz_min = spec.paper_tz_min;
    ibvs.paper_tz_max = spec.paper_tz_max;
    ibvs.paper_wz_max = spec.paper_wz_max;
    ibvs.paper_theta_pair = spec.paper_theta_pair;

    ibvs.mpc_horizon = spec.mpc_horizon;
    ibvs.mpc_q_weight = spec.mpc_q_weight;
    ibvs.mpc_r_weight = spec.mpc_r_weight;
    ibvs.mpc_slack_weight = spec.mpc_slack_weight;
    ibvs.mpc_v_max = spec.mpc_v_max;
    ibvs.mpc_w_max = spec.mpc_w_max;
    ibvs.mpc_fov_margin = spec.mpc_fov_margin;
    ibvs.mpc_solve_repeats = spec.mpc_solve_repeats;

    if strcmpi(spec.key, 'classical')
        % Keep Cheng/OG-VFVR on its own legacy front-end only.
        ibvs.legacy_fov_enabled = true;
        ibvs.pf_enabled = false;
        ibvs.paper_apf_enabled = false;
        ibvs.mpc_enabled = false;
    end

    ibvs.init();
    status = 0;
    for k = 1:numel(dt_seq)
        ibvs.Task_process_clcle = dt_seq(k);
        status_step = ibvs.step();
        status = status_step;
        if ~isempty(ibvs.history)
            latest_hist = ibvs.history(end);
            use_virtual_eval = method_uses_virtual_feature(spec.key);
            eval_target_error = history_eval_target_error(latest_hist, scene, use_virtual_eval);
            eval_boundary = history_eval_boundary(latest_hist, scene, use_virtual_eval);

            if eval_boundary <= 0
                status = -1;
            elseif eval_target_error <= scene.eterm
                status = 1;
            elseif status_step > 0
                status = 0;
            end
        end
        if status ~= 0
            break;
        end
    end

    result = empty_result();
    result.key = spec.key;
    result.name = spec.name;
    result.short_name = spec.short_name;
    result.use_nd = use_nd;
    result.variant = ternary(use_nd, 'withND', 'noND');
    result.status = status;
    result.history = ibvs.history;
    result.data = extract_history_data(ibvs.history, scene);
    result.metrics = compute_case_metrics(result.data, status, spec.key);
end


function dt_seq = make_dt_sequence(scene)
    dt_seq = repmat(scene.dt_ms, scene.max_iter, 1);
    if ~scene.use_random_dt
        return;
    end

    rng(scene.random_seed);
    jitter = randi([-5 5], scene.max_iter, 1);
    dt_seq = max(1, scene.dt_ms + jitter);
end


function data = extract_history_data(history, scene)
    data = struct();
    n = numel(history);
    data.n = n;
    data.f = [];
    data.real_f = [];
    data.e = [];
    data.en = [];
    data.control_e = [];
    data.control_en = [];
    data.target_en = [];
    data.real_target_en = [];
    data.jcond = [];
    data.raw_vel = [];
    data.exec_delta = [];
    data.exec_vel = [];
    data.dt = [];
    data.position_initial = scene.initial_t(:);
    data.position = [];
    data.position_path = scene.initial_t(:);
    data.boundary = [];
    data.real_boundary = [];
    data.lambda = [];

    if n == 0
        return;
    end

    nfeat2 = numel(history(1).f);
    data.f = zeros(nfeat2, n);
    data.real_f = zeros(nfeat2, n);
    data.e = zeros(nfeat2, n);
    data.en = zeros(1, n);
    data.control_e = zeros(nfeat2, n);
    data.control_en = zeros(1, n);
    data.target_en = zeros(1, n);
    data.real_target_en = zeros(1, n);
    data.jcond = zeros(1, n);
    data.raw_vel = zeros(6, n);
    data.exec_delta = zeros(6, n);
    data.exec_vel = zeros(6, n);
    data.dt = zeros(1, n);
    data.position = zeros(3, n);
    data.boundary = zeros(1, n);
    data.real_boundary = zeros(1, n);
    data.lambda = zeros(1, n);

    for k = 1:n
        data.f(:,k) = history(k).f(:);
        if isfield(history(k), 'real_f') && ~isempty(history(k).real_f)
            data.real_f(:,k) = history(k).real_f(:);
        else
            data.real_f(:,k) = history(k).f(:);
        end
        data.e(:,k) = history(k).e(:);
        data.en(k) = history(k).en;
        data.control_e(:,k) = data.e(:,k);
        if isfield(history(k), 's_ref') && ~isempty(history(k).s_ref)
            s_ref = history(k).s_ref(:);
            if numel(s_ref) == nfeat2
                data.control_e(:,k) = history(k).f(:) - s_ref;
            end
        end
        data.control_en(k) = norm(data.control_e(:,k));
        data.target_en(k) = norm(history(k).f(:) - scene.pd(:));
        data.real_target_en(k) = norm(data.real_f(:,k) - scene.pd(:));
        data.jcond(k) = history(k).jcond;
        data.raw_vel(:,k) = history(k).raw_vel(:);
        data.dt(k) = get_history_dt(history(k), scene);
        if isfield(history, 'executed_delta') && ~isempty(history(k).executed_delta)
            data.exec_delta(:,k) = history(k).executed_delta(:);
        else
            data.exec_delta(:,k) = history(k).vel(:);
        end
        data.exec_vel(:,k) = data.exec_delta(:,k) ./ max(data.dt(k), eps);
        data.position(:,k) = history(k).Tcam.t(:);
        uv = reshape(history(k).f, 2, []);
        data.boundary(k) = min([uv(1,:), uv(2,:), scene.npix(1)-uv(1,:), scene.npix(2)-uv(2,:)]);
        uv_real = reshape(data.real_f(:,k), 2, []);
        data.real_boundary(k) = min([uv_real(1,:), uv_real(2,:), scene.npix(1)-uv_real(1,:), scene.npix(2)-uv_real(2,:)]);
        if isfield(history, 'lambda') && ~isempty(history(k).lambda)
            data.lambda(k) = history(k).lambda;
        else
            data.lambda(k) = NaN;
        end
    end
    data.position_path = [data.position_initial, data.position];
end


function dt = get_history_dt(hist_item, scene)
    if isfield(hist_item, 'dt') && ~isempty(hist_item.dt)
        dt = hist_item.dt;
    else
        dt = 1 / scene.fps;
    end
end


function feat = history_real_feature(hist_item)
    if isfield(hist_item, 'real_f') && ~isempty(hist_item.real_f)
        feat = hist_item.real_f(:);
    else
        feat = hist_item.f(:);
    end
end


function tf = method_uses_virtual_feature(method_key)
    tf = strcmpi(method_key, 'classical');
end


function feat = history_eval_feature(hist_item, use_virtual)
    if use_virtual
        feat = hist_item.f(:);
    else
        feat = history_real_feature(hist_item);
    end
end


function err = history_eval_target_error(hist_item, scene, use_virtual)
    feat = history_eval_feature(hist_item, use_virtual);
    err = norm(feat(:) - scene.pd(:));
end


function boundary = history_eval_boundary(hist_item, scene, use_virtual)
    uv_eval = reshape(history_eval_feature(hist_item, use_virtual), 2, []);
    boundary = min([uv_eval(1,:), uv_eval(2,:), ...
        scene.npix(1) - uv_eval(1,:), scene.npix(2) - uv_eval(2,:)]);
end


function metrics = compute_case_metrics(data, status, method_key)
    metrics = struct( ...
        'status', status, 'steps', data.n, 'final_error', NaN, ...
        'initial_target_error', NaN, 'final_target_error', NaN, 'max_error', NaN, ...
        'max_target_error', NaN, 'mean_jcond', NaN, 'max_jcond', NaN, ...
        'max_velocity_norm', NaN, 'mean_velocity_norm', NaN, ...
        'max_increment_norm', NaN, 'mean_increment_norm', NaN, ...
        'min_boundary', NaN, 'fov_violations', 0, ...
        'min_real_boundary', NaN, 'real_fov_violations', 0, ...
        'initial_real_target_error', NaN, 'final_real_target_error', NaN, ...
        'failure_step', NaN, 'failure_reason', '', ...
        'final_x', NaN, 'final_y', NaN, 'final_z', NaN);

    if data.n == 0
        metrics.failure_reason = classify_status(status, data, method_key);
        return;
    end

    vel_norm = vecnorm(data.exec_vel, 2, 1);
    inc_norm = vecnorm(data.exec_delta, 2, 1);
    if method_uses_virtual_feature(method_key)
        eval_target_en = data.target_en;
        eval_boundary = data.boundary;
    elseif strcmpi(method_key, 'pf_ndrg')
        eval_target_en = data.en;
        eval_boundary = data.real_boundary;
    else
        eval_target_en = data.real_target_en;
        eval_boundary = data.real_boundary;
    end

    metrics.final_error = eval_target_en(end);
    metrics.initial_target_error = eval_target_en(1);
    metrics.final_target_error = eval_target_en(end);
    metrics.max_error = max(eval_target_en);
    metrics.max_target_error = max(eval_target_en);
    metrics.mean_jcond = mean(data.jcond, 'omitnan');
    metrics.max_jcond = max(data.jcond);
    metrics.max_velocity_norm = max(vel_norm);
    metrics.mean_velocity_norm = mean(vel_norm, 'omitnan');
    metrics.max_increment_norm = max(inc_norm);
    metrics.mean_increment_norm = mean(inc_norm, 'omitnan');
    metrics.min_boundary = min(eval_boundary);
    metrics.fov_violations = sum(eval_boundary <= 0);
    if method_uses_virtual_feature(method_key)
        metrics.min_real_boundary = NaN;
        metrics.real_fov_violations = NaN;
        metrics.initial_real_target_error = NaN;
        metrics.final_real_target_error = NaN;
    else
        metrics.min_real_boundary = min(data.real_boundary);
        metrics.real_fov_violations = sum(data.real_boundary <= 0);
        metrics.initial_real_target_error = data.real_target_en(1);
        metrics.final_real_target_error = data.real_target_en(end);
    end
    if status < 0
        first_violation = find(eval_boundary <= 0, 1, 'first');
        if isempty(first_violation)
            metrics.failure_step = data.n;
        else
            metrics.failure_step = first_violation;
        end
    end
    metrics.failure_reason = classify_status(status, data, method_key);
    metrics.final_x = data.position(1,end);
    metrics.final_y = data.position(2,end);
    metrics.final_z = data.position(3,end);
end


function reason = classify_status(status, data, method_key)
    if status > 0
        reason = 'converged';
    elseif status == 0
        reason = 'max_iter_not_converged';
    elseif data.n == 0
        reason = 'controller_failure_before_history';
    elseif method_uses_virtual_feature(method_key) && min(data.boundary) <= 0
        reason = 'fov_violation_virtual_feature_left_image';
    elseif ~method_uses_virtual_feature(method_key) && min(data.real_boundary) <= 0
        reason = 'fov_violation_feature_left_image';
    else
        reason = 'controller_failure_non_fov';
    end
end


function plot_3d_trajectories(results, scene, out_dir, suffix, label)
    target_result = select_target_3d_result(results);

    fig = figure('Name', ['Fig1 3D camera trajectory ' suffix], ...
        'Color', 'w', 'Position', [100 100 520 420]);
    ax = axes('Parent', fig, 'Position', [0.13 0.13 0.78 0.78]);
    plot_target_3d_axis(ax, target_result, target_3d_limits(), true);
    save_figure(fig, out_dir, ['fig01_3d_trajectories_' suffix]);

    compare_fig = figure('Name', ['Fig1 3D camera trajectories comparison ' suffix], ...
        'Color', 'w', 'Position', [80 80 1100 820]);
    figure(compare_fig);
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    [xlim_all, ylim_all, zlim_all] = common_3d_limits(results, scene);
    compare_limits = struct('x', xlim_all, 'y', ylim_all, 'z', zlim_all, ...
        'xtick', [], 'ytick', [], 'ztick', []);
    for i = 1:numel(results)
        ax = nexttile;
        plot_target_3d_axis(ax, results(i), compare_limits, false);
        set_axis_with_panel_label(ax, '$x$ (m)', panel_label(i), 'latex');
    end
    figure(compare_fig);
    save_figure(compare_fig, out_dir, ['fig01_3d_trajectories_compare_' suffix]);
end


function result = select_target_3d_result(results)
    idx = find(strcmpi({results.key}, 'pf_ndrg'), 1);
    if isempty(idx)
        has_data = arrayfun(@(r) isfield(r, 'data') && isfield(r.data, 'n') && r.data.n > 0, results);
        idx = find(has_data, 1, 'last');
    end
    if isempty(idx)
        idx = 1;
    end
    result = results(idx);
end


function limits = target_3d_limits()
    limits = struct( ...
        'x', [-2.5 0.05], ...
        'y', [-1.1 0.1], ...
        'z', [-4.05 -0.85], ...
        'xtick', [-2 -1 0], ...
        'ytick', [-1 -0.5 0], ...
        'ztick', [-4 -3 -2 -1]);
end


function plot_target_3d_axis(ax, result, limits, show_legend)
    hold(ax, 'on');
    grid(ax, 'on');
    box(ax, 'on');

    data = result.data;
    if data.n > 0
        path = data.position_path;
        h_traj = plot3(ax, path(1,:), path(2,:), path(3,:), ...
            'b-', 'LineWidth', 2.1, 'DisplayName', 'Trajectory');
        h_start = plot3(ax, data.position_initial(1), data.position_initial(2), data.position_initial(3), ...
            'o', 'Color', [0 1 0], 'MarkerFaceColor', [0 1 0], ...
            'MarkerSize', 7, 'LineWidth', 1.0, 'DisplayName', 'Start');
        h_end = plot3(ax, data.position(1,end), data.position(2,end), data.position(3,end), ...
            'o', 'Color', [1 0 0], 'MarkerFaceColor', [1 0 0], ...
            'MarkerSize', 7, 'LineWidth', 1.0, 'DisplayName', 'End');
    else
        h_traj = plot3(ax, NaN, NaN, NaN, 'b-', 'LineWidth', 2.1, 'DisplayName', 'Trajectory');
        h_start = plot3(ax, NaN, NaN, NaN, 'o', 'Color', [0 1 0], ...
            'MarkerFaceColor', [0 1 0], 'MarkerSize', 7, 'DisplayName', 'Start');
        h_end = plot3(ax, NaN, NaN, NaN, 'o', 'Color', [1 0 0], ...
            'MarkerFaceColor', [1 0 0], 'MarkerSize', 7, 'DisplayName', 'End');
    end

    xlabel(ax, '$x$ (m)', 'Interpreter', 'latex', 'FontSize', 11);
    ylabel(ax, '$y$ (m)', 'Interpreter', 'latex', 'FontSize', 11);
    zlabel(ax, '$z$ (m)', 'Interpreter', 'latex', 'FontSize', 11);
    xlim(ax, limits.x);
    ylim(ax, limits.y);
    zlim(ax, limits.z);
    if ~isempty(limits.xtick), xticks(ax, limits.xtick); end
    if ~isempty(limits.ytick), yticks(ax, limits.ytick); end
    if ~isempty(limits.ztick), zticks(ax, limits.ztick); end
    view(ax, -45, 18);
    pbaspect(ax, [1.2 1.0 0.75]);
    axis(ax, 'vis3d');

    set(ax, 'FontName', 'Times New Roman', 'FontSize', 10, ...
        'LineWidth', 0.8, 'GridColor', [0.84 0.84 0.84], ...
        'GridAlpha', 0.65, 'TickDir', 'out', 'Color', 'w');

    if show_legend
        lgd = legend(ax, [h_traj h_start h_end], {'Trajectory', 'Start', 'End'}, ...
            'Location', 'northwest', 'Box', 'on', 'Color', 'w', ...
            'EdgeColor', [0.25 0.25 0.25], 'Interpreter', 'none', 'FontSize', 9);
        try
            lgd.ItemTokenSize = [28 14];
        catch
        end
        try
            lgd.Units = 'normalized';
            lgd_pos = lgd.Position;
            lgd_pos(1) = 0.17;
            lgd_pos(2) = 0.70;
            lgd.Position = lgd_pos;
        catch
        end
    end
end


function [xlim_all, ylim_all, zlim_all] = common_3d_limits(results, scene)
    if isfield(scene, 'plot3d_xlim') && isfield(scene, 'plot3d_ylim') && isfield(scene, 'plot3d_zlim')
        xlim_all = scene.plot3d_xlim;
        ylim_all = scene.plot3d_ylim;
        zlim_all = scene.plot3d_zlim;
        return;
    end

    pts = scene.initial_t(:);
    for i = 1:numel(results)
        data = results(i).data;
        if isfield(data, 'position_path') && ~isempty(data.position_path)
            pts = [pts data.position_path]; %#ok<AGROW>
        elseif isfield(data, 'position') && ~isempty(data.position)
            pts = [pts data.position]; %#ok<AGROW>
        end
    end
    xlim_all = padded_limits(pts(1,:));
    ylim_all = padded_limits(pts(2,:));
    zlim_all = padded_limits(pts(3,:));
end


function lim = padded_limits(values)
    values = values(isfinite(values));
    if isempty(values)
        lim = [-1 1];
        return;
    end
    lo = min(values);
    hi = max(values);
    span = hi - lo;
    if span < 1e-6
        span = max(1, abs(hi));
    end
    pad = 0.08 * span;
    lim = [lo - pad, hi + pad];
end


function plot_image_feature_trajectories(results, scene, out_dir, suffix, label)
    fig = figure('Name', ['Fig2 image feature trajectories ' suffix], ...
        'Color', 'w', 'Position', [80 80 1240 820]);
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    for i = 1:numel(results)
        ax = nexttile;
        plot_feature_panel(ax, results(i), scene, panel_label(i));
    end
    save_figure(fig, out_dir, ['fig02_image_feature_trajectories_' suffix]);
end


function plot_feature_panel(ax, result, scene, panel_label)
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    set(ax, 'YDir', 'reverse', 'FontName', 'Times New Roman', 'FontSize', 10, ...
        'LineWidth', 0.9, 'GridAlpha', 0.18, 'GridColor', [0 0 0]);
    xlim(ax, [0 scene.npix(1)]);
    ylim(ax, [0 scene.npix(2)]);
    ylabel(ax, 'v (pixels)');
    set_axis_with_panel_label(ax, 'u (pixels)', panel_label);

    desired_color = [0.92 0.10 0.10];
    start_color = [0.00 0.35 0.16];
    traj_color = lines(max(1, size(scene.pd, 2)));

    pd_closed = [scene.pd scene.pd(:,1)];
    plot(ax, pd_closed(1,:), pd_closed(2,:), ':', 'Color', desired_color, ...
        'LineWidth', 1.0, 'DisplayName', 'Desired contour');
    plot(ax, scene.pd(1,:), scene.pd(2,:), 'r*', 'LineWidth', 1.0, ...
        'MarkerSize', 6.5, 'DisplayName', 'Desired');

    data = result.data;
    feat_hist = feature_history_for_plot(result);
    if data.n > 0 && ~isempty(feat_hist)
        npts = size(feat_hist, 1) / 2;
        traj = reshape(feat_hist, 2, npts, []);
        uv0 = traj(:,:,1);
        uvf = traj(:,:,end);
        uv0_closed = [uv0 uv0(:,1)];

        plot(ax, uv0_closed(1,:), uv0_closed(2,:), '--', 'Color', start_color, ...
            'LineWidth', 1.0, 'DisplayName', 'Initial contour');
        plot(ax, uv0(1,:), uv0(2,:), 'o', 'Color', start_color, ...
            'MarkerFaceColor', [0.70 0.84 0.70], 'MarkerSize', 5.5, ...
            'LineWidth', 1.0, 'DisplayName', 'Initial feature');

        for p = 1:npts
            u = squeeze(traj(1,p,:));
            v = squeeze(traj(2,p,:));
            plot(ax, u, v, '-', 'Color', traj_color(p,:), 'LineWidth', 1.35, ...
                'DisplayName', sprintf('p%d trajectory', p));
        end

        if method_uses_virtual_feature(result.key)
            text(ax, 0.03, 0.06, 'Cheng panel: solid lines are virtual f used by the controller', ...
                'Units', 'normalized', 'FontName', 'Times New Roman', ...
                'FontSize', 8.5, 'Color', [0.55 0.18 0.04], ...
                'BackgroundColor', [1 1 1], 'Margin', 2, ...
                'Interpreter', 'none');
        end

        plot(ax, uvf(1,:), uvf(2,:), 's', 'Color', [0.10 0.10 0.10], ...
            'MarkerFaceColor', [0.10 0.10 0.10], 'MarkerSize', 3.6, ...
            'LineWidth', 0.9, 'DisplayName', 'Final feature');
    else
        plot(ax, scene.p0(1,:), scene.p0(2,:), 'o', 'Color', start_color, ...
            'MarkerFaceColor', [0.70 0.84 0.70], 'MarkerSize', 5.5, ...
            'LineWidth', 1.0, 'DisplayName', 'Initial feature');
    end
end


function plot_jacobian_condition(results, out_dir, suffix, label)
    fig = figure('Name', ['Fig3 Jacobian condition ' suffix], ...
        'Color', 'w', 'Position', [80 80 1220 780]);
    rects = two_by_two_axes_rects();
    for i = 1:numel(results)
        ax = axes('Parent', fig, 'Units', 'normalized', 'Position', rects(i,:));
        hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
        set(ax, 'FontName', 'Times New Roman', 'FontSize', 10, ...
            'LineWidth', 0.9, 'GridAlpha', 0.18, 'GridColor', [0 0 0]);
        data = results(i).data;
        if data.n > 0
            plot(ax, 1:data.n, data.jcond, 'LineWidth', 1.45, 'Color', [0.11 0.38 0.84]);
        end
        ylabel(ax, 'Jacobian condition number');
        set_axis_with_panel_label(ax, 'Number of iterations', ...
            panel_label(i));
    end
    save_figure(fig, out_dir, ['fig03_jacobian_condition_' suffix]);
end


function plot_feature_error(results, scene, out_dir, suffix, label)
    fig = figure('Name', ['Fig4 feature error ' suffix], ...
        'Color', 'w', 'Position', [80 80 1220 820]);
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    legend_panel = 1;
    if strcmpi(suffix, 'noND')
        legend_panel = 2;
    end
    for i = 1:numel(results)
        ax = nexttile;
        plot_feature_error_panel(ax, results(i), scene, panel_label(i), i == legend_panel);
    end
    save_figure(fig, out_dir, ['fig04_feature_error_' suffix]);
end


function plot_feature_error_panel(ax, result, scene, panel_label, show_legend)
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 10, ...
        'LineWidth', 0.9, 'GridAlpha', 0.18, 'GridColor', [0 0 0]);

    data = result.data;
    err_mat = desired_feature_error_matrix(result, scene);
    if ~isempty(err_mat)
        npts = size(err_mat, 1) / 2;
        colors = lines(npts);
        for p = 1:npts
            plot(ax, 1:data.n, err_mat(2*p-1,:), '-', 'Color', colors(p,:), ...
                'LineWidth', 1.45, 'DisplayName', sprintf('e_{u%d}', p));
            plot(ax, 1:data.n, err_mat(2*p,:), '--', 'Color', colors(p,:), ...
                'LineWidth', 1.45, 'DisplayName', sprintf('e_{v%d}', p));
        end
    end

    ylabel(ax, 'Feature error (pixels)');
    if strcmpi(result.key, 'pf_ndrg')
        text(ax, 0.03, 0.92, 'PF-NDRG: error = current feature - virtual target', ...
            'Units', 'normalized', 'FontName', 'Times New Roman', ...
            'FontSize', 8.3, 'Color', [0.10 0.28 0.55], ...
            'BackgroundColor', [1 1 1], 'Margin', 2, 'Interpreter', 'none');
    end
    set_axis_with_panel_label(ax, 'Number of iterations', panel_label);
    if show_legend
        lgd = legend(ax, 'Location', 'northeast', 'Interpreter', 'tex', ...
            'FontSize', 8.2, 'NumColumns', 4, 'Box', 'on');
        try
            lgd.ItemTokenSize = [18 10];
        catch
        end
    end
end


function plot_control_feature_error(results, scene, out_dir, suffix, label)
    fig = figure('Name', ['Fig4b control feature error ' suffix], ...
        'Color', 'w', 'Position', [80 80 1220 820]);
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    legend_panel = 1;
    if strcmpi(suffix, 'noND')
        legend_panel = 2;
    end
    for i = 1:numel(results)
        ax = nexttile;
        plot_control_feature_error_panel(ax, results(i), scene, panel_label(i), i == legend_panel);
    end
    save_figure(fig, out_dir, ['fig04b_feature_error_control_e_' suffix]);
end


function plot_control_feature_error_panel(ax, result, scene, panel_label, show_legend)
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 10, ...
        'LineWidth', 0.9, 'GridAlpha', 0.18, 'GridColor', [0 0 0]);

    data = result.data;
    err_mat = control_feature_error_matrix(result, scene);
    if ~isempty(err_mat)
        npts = size(err_mat, 1) / 2;
        colors = lines(npts);
        for p = 1:npts
            plot(ax, 1:data.n, err_mat(2*p-1,:), '-', 'Color', colors(p,:), ...
                'LineWidth', 1.45, 'DisplayName', sprintf('e_{u%d}', p));
            plot(ax, 1:data.n, err_mat(2*p,:), '--', 'Color', colors(p,:), ...
                'LineWidth', 1.45, 'DisplayName', sprintf('e_{v%d}', p));
        end
    end

    ylabel(ax, 'Control feature error (pixels)');
    if strcmpi(result.key, 'pf_ndrg')
        text(ax, 0.03, 0.92, 'PF-NDRG: error = current feature - updated virtual target', ...
            'Units', 'normalized', 'FontName', 'Times New Roman', ...
            'FontSize', 8.3, 'Color', [0.10 0.28 0.55], ...
            'BackgroundColor', [1 1 1], 'Margin', 2, 'Interpreter', 'none');
    end
    set_axis_with_panel_label(ax, 'Number of iterations', panel_label);
    if show_legend
        lgd = legend(ax, 'Location', 'northeast', 'Interpreter', 'tex', ...
            'FontSize', 8.2, 'NumColumns', 4, 'Box', 'on');
        lgd.ItemTokenSize = [18 10];
    end
end


function plot_executed_displacement(results, out_dir, suffix, label)
    fig = figure('Name', ['Fig5 executed displacement ' suffix], ...
        'Color', 'w', 'Position', [40 40 1500 1120]);
    tile_rects = four_tile_rects();
    for i = 1:numel(results)
        tile_pos = tile_rects(i,:);
        caption_h = min(0.055, max(0.040, 0.11 * tile_pos(4)));
        panel_rect = [tile_pos(1), tile_pos(2) + caption_h, tile_pos(3), tile_pos(4) - caption_h];
        plot_component_axes_in_rect(fig, panel_rect, results(i).data.exec_delta, ...
            increment_component_labels(), [0.10 0.36 0.85]);
        annotation(fig, 'textbox', [tile_pos(1), tile_pos(2), tile_pos(3), caption_h], ...
            'String', panel_label(i), ...
            'Interpreter', 'none', 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'EdgeColor', 'none', ...
            'FontName', 'Times New Roman', 'FontSize', 11);
    end

    save_figure(fig, out_dir, ['fig05_executed_displacement_' suffix]);
end


function plot_executed_displacement_nd_compare(results, out_dir)
    fig = figure('Name', 'Fig5 executed displacement Baseline vs ND method', ...
        'Color', 'w', 'Position', [40 40 1500 1120]);
    add_global_pair_key(fig);

    tile_rects = four_tile_rects();
    for i = 1:size(results, 1)
        tile_pos = tile_rects(i,:);
        caption_h = min(0.055, max(0.040, 0.11 * tile_pos(4)));
        panel_rect = [tile_pos(1), tile_pos(2) + caption_h, tile_pos(3), tile_pos(4) - caption_h];
        plot_component_pair_axes_in_rect(fig, panel_rect, ...
            results(i,1).data.exec_delta, results(i,2).data.exec_delta, ...
            increment_component_labels());
        annotation(fig, 'textbox', [tile_pos(1), tile_pos(2), tile_pos(3), caption_h], ...
            'String', panel_label(i), ...
            'Interpreter', 'none', 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'EdgeColor', 'none', ...
            'FontName', 'Times New Roman', 'FontSize', 11);
    end

    save_figure(fig, out_dir, 'fig05_executed_displacement_baseline_vs_nd');
end


function add_global_pair_key(fig)
    y = 0.965;
    annotation(fig, 'line', [0.740 0.770], [y y], ...
        'Color', [0.10 0.32 0.90], 'LineWidth', 1.7);
    annotation(fig, 'textbox', [0.775 0.948 0.075 0.035], ...
        'String', 'Baseline', 'Interpreter', 'none', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle', 'FontName', 'Times New Roman', ...
        'FontSize', 9);
    annotation(fig, 'line', [0.850 0.880], [y y], ...
        'Color', [0.92 0.12 0.12], 'LineWidth', 1.7);
    annotation(fig, 'textbox', [0.885 0.948 0.100 0.035], ...
        'String', 'ND method', 'Interpreter', 'none', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle', 'FontName', 'Times New Roman', ...
        'FontSize', 9);
end


function plot_no_nd_end_effector_velocity(results, out_dir)
    fig = figure('Name', 'No-ND end-effector velocity by method', ...
        'Color', 'w', 'Position', [80 80 1240 820]);
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    labels = velocity_component_labels();
    colors = lines(6);
    line_styles = {'-', '-', '-', '--', '--', '--'};
    for i = 1:numel(results)
        ax = nexttile;
        hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
        set(ax, 'FontName', 'Times New Roman', 'FontSize', 10, ...
            'LineWidth', 0.9, 'GridAlpha', 0.18, 'GridColor', [0 0 0]);

        data = results(i).data;
        if ~isempty(data) && data.n > 0 && isfield(data, 'exec_vel') && ~isempty(data.exec_vel)
            t = 1:data.n;
            for c = 1:min(6, size(data.exec_vel, 1))
                plot(ax, t, data.exec_vel(c,:), line_styles{c}, ...
                    'Color', colors(c,:), 'LineWidth', 1.25, ...
                    'DisplayName', labels{c});
            end
        end

        ylabel(ax, 'End-effector velocity');
        set_axis_with_panel_label(ax, 'Number of iterations', ...
            panel_label(i));
        if i == 1
            legend(ax, 'Location', 'northeast', 'Interpreter', 'tex', ...
                'FontSize', 8.2, 'NumColumns', 2, 'Box', 'on');
        end
    end

    save_figure(fig, out_dir, 'fig09_end_effector_velocity_noND');
end


function plot_nd_pair_feature_error(results, scene, out_dir)
    fig = figure('Name', 'Fig6 ND vs no-ND method-feature error', ...
        'Color', 'w', 'Position', [80 80 1220 820]);
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    for i = 1:size(results, 1)
        ax = nexttile;
        hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
        set(ax, 'FontName', 'Times New Roman', 'FontSize', 10, ...
            'LineWidth', 0.9, 'GridAlpha', 0.18, 'GridColor', [0 0 0]);

        y0 = error_norm_against_target(results(i,1), scene);
        y1 = error_norm_against_target(results(i,2), scene);
        plot_pair_series(ax, y0, 'Baseline', '-', [0.10 0.32 0.90]);
        plot_pair_series(ax, y1, 'ND method', '-', [0.92 0.12 0.12]);

        ylabel(ax, '||p-p^*||_2 (pixels)');
        set_axis_with_panel_label(ax, 'Number of iterations', ...
            panel_label(i));
        if i == 1
            legend(ax, 'Location', 'northeast', 'Interpreter', 'none', 'FontSize', 8.5);
        end
    end
    save_figure(fig, out_dir, 'fig06_nd_vs_no_nd_method_feature_error');
end


function plot_nd_pair_increment_norm(results, out_dir)
    comp_labels = increment_component_labels();
    for i = 1:size(results, 1)
        fig = figure('Name', ['Fig7 ND vs no-ND displacement ' results(i,1).key], ...
            'Color', 'w', 'Position', [60 60 980 780]);
        tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
        for c = 1:6
            ax = nexttile;
            hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
            set(ax, 'FontName', 'Times New Roman', 'FontSize', 10, ...
                'LineWidth', 0.9, 'GridAlpha', 0.18, 'GridColor', [0 0 0]);
            plot_pair_series(ax, component_series(results(i,1).data.exec_delta, c), ...
                'Baseline', '-', [0.10 0.32 0.90]);
            plot_pair_series(ax, component_series(results(i,2).data.exec_delta, c), ...
                'ND method', '-', [0.92 0.12 0.12]);
            ylabel(ax, comp_labels{c});
            if c == 1
                legend(ax, 'Location', 'northeast', 'Interpreter', 'none', 'FontSize', 8.5);
            end
            x_text = ternary(c > 4, 'Number of iterations', '');
            set_axis_with_bottom_panel_label(ax, x_text, panel_label(c));
        end
        save_figure(fig, out_dir, ['fig07_nd_vs_no_nd_displacement_' results(i,1).key]);
    end
end


function plot_cheng_virtual_fov_diagnostics(results, scene, out_dir)
    row = find(arrayfun(@(r) strcmpi(r.key, 'classical'), results(:,1)), 1, 'first');
    if isempty(row)
        return;
    end

    cheng_results = results(row,:);
    fig = figure('Name', 'Fig8 Cheng virtual FOV diagnostics', ...
        'Color', 'w', 'Position', [70 70 1320 920]);
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax = nexttile;
    plot_cheng_virtual_feature_panel(ax, cheng_results(1), scene, ...
        panel_label(1));

    ax = nexttile;
    plot_cheng_virtual_feature_panel(ax, cheng_results(2), scene, ...
        panel_label(2));

    ax = nexttile;
    plot_cheng_shift_panel(ax, cheng_results, panel_label(3));

    ax = nexttile;
    plot_cheng_boundary_panel(ax, cheng_results, panel_label(4));

    save_figure(fig, out_dir, 'fig08_cheng_virtual_fov_diagnostics');
end


function plot_cheng_virtual_feature_panel(ax, result, scene, panel_label)
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    set(ax, 'YDir', 'reverse', 'FontName', 'Times New Roman', 'FontSize', 10, ...
        'LineWidth', 0.9, 'GridAlpha', 0.18, 'GridColor', [0 0 0]);
    xlim(ax, [0 scene.npix(1)]);
    ylim(ax, [0 scene.npix(2)]);
    ylabel(ax, 'v (pixels)');
    set_axis_with_panel_label(ax, 'u (pixels)', panel_label);

    pd_closed = [scene.pd scene.pd(:,1)];
    plot(ax, pd_closed(1,:), pd_closed(2,:), ':', 'Color', [0.92 0.10 0.10], ...
        'LineWidth', 1.0, 'DisplayName', 'Desired contour');
    plot(ax, scene.pd(1,:), scene.pd(2,:), 'r*', 'LineWidth', 1.0, ...
        'MarkerSize', 7.5, 'DisplayName', 'Desired');

    data = result.data;
    if isempty(data) || data.n == 0 || isempty(data.f)
        return;
    end

    npts = size(data.f, 1) / 2;
    virtual_traj = reshape(data.f, 2, npts, []);
    colors = lines(npts);

    uv0 = virtual_traj(:,:,1);
    uv0_closed = [uv0 uv0(:,1)];
    plot(ax, uv0_closed(1,:), uv0_closed(2,:), '--', 'Color', [0.00 0.35 0.16], ...
        'LineWidth', 1.0, 'DisplayName', 'Initial contour');
    plot(ax, uv0(1,:), uv0(2,:), 'o', 'Color', [0.00 0.35 0.16], ...
        'MarkerFaceColor', [0.70 0.84 0.70], 'MarkerSize', 5.5, ...
        'LineWidth', 1.0, 'DisplayName', 'Initial virtual f');

    for p = 1:npts
        virtual_u = squeeze(virtual_traj(1,p,:));
        virtual_v = squeeze(virtual_traj(2,p,:));
        plot(ax, virtual_u, virtual_v, '-', 'Color', colors(p,:), 'LineWidth', 1.35, ...
            'DisplayName', sprintf('p%d virtual f', p));
    end

    uvf = virtual_traj(:,:,end);
    plot(ax, uvf(1,:), uvf(2,:), 's', 'Color', [0.10 0.10 0.10], ...
        'MarkerFaceColor', [0.10 0.10 0.10], 'MarkerSize', 4.0, ...
        'LineWidth', 0.9, 'DisplayName', 'Final virtual f');
    text(ax, 0.03, 0.06, 'solid lines: virtual f used by Cheng controller', ...
        'Units', 'normalized', 'FontName', 'Times New Roman', ...
        'FontSize', 8.5, 'Color', [0.55 0.18 0.04], ...
        'BackgroundColor', [1 1 1], 'Margin', 2, 'Interpreter', 'none');
end


function plot_cheng_shift_panel(ax, cheng_results, panel_label)
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 10, ...
        'LineWidth', 0.9, 'GridAlpha', 0.18, 'GridColor', [0 0 0]);
    colors = [0.10 0.32 0.90; 0.92 0.12 0.12];
    for v = 1:numel(cheng_results)
        [active, ~, shift] = legacy_fov_shift_series(cheng_results(v).history);
        t = 1:numel(active);
        shift_norm = vecnorm(shift, 2, 1);
        variant_label = nd_variant_legend_label(cheng_results(v).variant);
        plot(ax, t, shift_norm, '-', 'Color', colors(v,:), 'LineWidth', 1.35, ...
            'DisplayName', [variant_label ' shift norm']);
        if any(active)
            scatter(ax, t(active), shift_norm(active), 13, colors(v,:), 'filled', ...
                'MarkerFaceAlpha', 0.45, 'DisplayName', [variant_label ' trigger']);
        end
    end
    ylabel(ax, 'Global virtual-FOV shift norm (pixels)');
    set_axis_with_panel_label(ax, 'Number of iterations', panel_label);
    legend(ax, 'Location', 'northeast', 'Interpreter', 'none', 'FontSize', 8.5);
end


function plot_cheng_boundary_panel(ax, cheng_results, panel_label)
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    set(ax, 'FontName', 'Times New Roman', 'FontSize', 10, ...
        'LineWidth', 0.9, 'GridAlpha', 0.18, 'GridColor', [0 0 0]);
    colors = [0.10 0.32 0.90; 0.92 0.12 0.12];
    for v = 1:numel(cheng_results)
        data = cheng_results(v).data;
        if isempty(data) || data.n == 0
            continue;
        end
        t = 1:data.n;
        variant_label = nd_variant_legend_label(cheng_results(v).variant);
        plot(ax, t, data.boundary, '-', 'Color', colors(v,:), ...
            'LineWidth', 1.15, 'DisplayName', [variant_label ' virtual f boundary']);
    end
    yline(ax, 0, 'k:', 'LineWidth', 1.0, 'DisplayName', 'image boundary');
    ylabel(ax, 'Minimum boundary margin (pixels)');
    set_axis_with_panel_label(ax, 'Number of iterations', panel_label);
    legend(ax, 'Location', 'southeast', 'Interpreter', 'none', 'FontSize', 8.5);
end


function [active, trigger_idx, shift] = legacy_fov_shift_series(history)
    n = numel(history);
    active = false(1, n);
    trigger_idx = nan(1, n);
    shift = zeros(2, n);
    for k = 1:n
        if isfield(history(k), 'legacy_fov_info') && ~isempty(history(k).legacy_fov_info)
            info = history(k).legacy_fov_info;
            if isfield(info, 'active') && ~isempty(info.active)
                active(k) = logical(info.active);
            end
            if isfield(info, 'index') && ~isempty(info.index)
                trigger_idx(k) = info.index;
            end
            if isfield(info, 'shift') && ~isempty(info.shift)
                shift(:,k) = info.shift(:);
            end
        end
    end
end


function plot_pair_series(ax, y, label_text, line_style, color_value)
    if isempty(y)
        return;
    end
    plot(ax, 1:numel(y), y, line_style, 'LineWidth', 1.45, ...
        'Color', color_value, 'DisplayName', label_text);
end


function label_text = nd_variant_legend_label(variant_key)
    if strcmpi(variant_key, 'noND') || strcmpi(variant_key, 'without ND')
        label_text = 'Baseline';
    elseif strcmpi(variant_key, 'withND') || strcmpi(variant_key, 'with ND')
        label_text = 'ND method';
    else
        label_text = variant_key;
    end
end


function feat_hist = feature_history_for_plot(result)
    feat_hist = [];
    data = result.data;
    if isempty(data)
        return;
    end
    if method_uses_virtual_feature(result.key) && isfield(data, 'f') && ~isempty(data.f)
        feat_hist = data.f;
    elseif isfield(data, 'real_f') && ~isempty(data.real_f) && any(isfinite(data.real_f(:)))
        feat_hist = data.real_f;
    elseif isfield(data, 'f') && ~isempty(data.f)
        feat_hist = data.f;
    end
end


function err_mat = desired_feature_error_matrix(result, scene)
    data = result.data;
    if strcmpi(result.key, 'pf_ndrg') && ~isempty(data) && isfield(data, 'e') && ~isempty(data.e)
        % PF-NDRG defines the plotted feature error as current feature minus
        % its virtual reference, not current feature minus the final target.
        err_mat = data.e;
        return;
    end

    feat_hist = feature_history_for_plot(result);
    if isempty(feat_hist)
        err_mat = [];
        return;
    end
    err_mat = feat_hist - repmat(scene.pd(:), 1, size(feat_hist, 2));
end


function err_mat = control_feature_error_matrix(result, scene)
    data = result.data;
    if strcmpi(result.key, 'pf_ndrg') && ~isempty(data) && ...
            isfield(data, 'control_e') && ~isempty(data.control_e)
        err_mat = data.control_e;
        return;
    end

    err_mat = desired_feature_error_matrix(result, scene);
end


function y = error_norm_against_target(result, scene)
    err_mat = desired_feature_error_matrix(result, scene);
    if isempty(err_mat)
        y = [];
        return;
    end
    y = vecnorm(err_mat, 2, 1);
end


function labels = increment_component_labels()
    labels = {'\Delta p_x (m)', '\Delta p_y (m)', '\Delta p_z (m)', ...
        '\Delta \theta_x (rad)', '\Delta \theta_y (rad)', '\Delta \theta_z (rad)'};
end


function labels = velocity_component_labels()
    labels = {'v_x', 'v_y', 'v_z', '\omega_x', '\omega_y', '\omega_z'};
end


function y = component_series(data_mat, idx)
    if isempty(data_mat) || size(data_mat, 1) < idx
        y = [];
    else
        y = data_mat(idx,:);
    end
end


function set_axis_with_panel_label(ax, primary_label, panel_label, interpreter)
    if nargin < 2 || isempty(primary_label)
        primary_label = '';
    end
    if nargin < 3 || isempty(panel_label)
        panel_label = '';
    end
    if nargin < 4 || isempty(interpreter)
        interpreter = 'none';
    end
    if isempty(panel_label)
        xlabel(ax, primary_label, 'Interpreter', interpreter);
    else
        xlabel(ax, {primary_label, panel_label}, 'Interpreter', interpreter);
    end
end


function label_text = panel_label(index)
    label_text = sprintf('(%c)', char('a' + index - 1));
end


function set_axis_with_bottom_panel_label(ax, primary_label, panel_label)
    if nargin < 2 || isempty(primary_label)
        primary_label = '';
    end
    if nargin < 3 || isempty(panel_label)
        panel_label = '';
    end

    xlabel(ax, primary_label, 'Interpreter', 'none');
    if isempty(panel_label)
        return;
    end

    y_pos = ternary(isempty(primary_label), -0.17, -0.33);
    text(ax, 0.5, y_pos, panel_label, ...
        'Units', 'normalized', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 11, ...
        'Interpreter', 'none', ...
        'Clipping', 'off');
end


function rects = two_by_two_axes_rects()
    margin_left = 0.080;
    margin_right = 0.035;
    margin_bottom = 0.095;
    margin_top = 0.045;
    gap_x = 0.070;
    gap_y = 0.090;

    ax_w = (1 - margin_left - margin_right - gap_x) / 2;
    ax_h = (1 - margin_bottom - margin_top - gap_y) / 2;
    rects = [
        margin_left,                   margin_bottom + ax_h + gap_y, ax_w, ax_h;
        margin_left + ax_w + gap_x,    margin_bottom + ax_h + gap_y, ax_w, ax_h;
        margin_left,                   margin_bottom,                ax_w, ax_h;
        margin_left + ax_w + gap_x,    margin_bottom,                ax_w, ax_h
    ];
end


function rects = four_tile_rects()
    margin_left = 0.055;
    margin_right = 0.035;
    margin_bottom = 0.025;
    margin_top = 0.050;
    gap_x = 0.040;
    gap_y = 0.115;

    tile_w = (1 - margin_left - margin_right - gap_x) / 2;
    tile_h = (1 - margin_top - margin_bottom - gap_y) / 2;

    rects = [
        margin_left,                     margin_bottom + tile_h + gap_y, tile_w, tile_h;
        margin_left + tile_w + gap_x,    margin_bottom + tile_h + gap_y, tile_w, tile_h;
        margin_left,                     margin_bottom,                  tile_w, tile_h;
        margin_left + tile_w + gap_x,    margin_bottom,                  tile_w, tile_h
    ];
end


function plot_component_axes_in_rect(fig, panel_rect, data_mat, comp_labels, series_color)
    margin_x = 0.015;
    margin_y_bottom = 0.030;
    margin_y_top = 0.006;
    gap_x = 0.060;
    gap_y = 0.036;

    ax_w = (panel_rect(3) - 2 * margin_x - gap_x) / 2;
    ax_h = (panel_rect(4) - margin_y_bottom - margin_y_top - 2 * gap_y) / 3;

    for row = 1:3
        for col = 1:2
            c = (row - 1) * 2 + col;
            pos_x = panel_rect(1) + margin_x + (col - 1) * (ax_w + gap_x);
            pos_y = panel_rect(2) + panel_rect(4) - margin_y_top - row * ax_h - (row - 1) * gap_y;
            ax = axes('Parent', fig, 'Units', 'normalized', 'Position', [pos_x pos_y ax_w ax_h]);
            hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
            set(ax, 'FontName', 'Times New Roman', 'FontSize', 8.5, ...
                'LineWidth', 0.8, 'GridAlpha', 0.15, 'GridColor', [0 0 0]);
            if ~isempty(data_mat) && size(data_mat, 1) >= c
                plot(ax, 1:size(data_mat, 2), data_mat(c,:), 'LineWidth', 1.1, 'Color', series_color);
            end
            ylabel(ax, comp_labels{c}, 'Interpreter', 'tex');
            set_axis_with_panel_label(ax, ternary(row == 3, 'Number of iterations', ''), '');
        end
    end
end


function plot_component_pair_axes_in_rect(fig, panel_rect, baseline_mat, nd_mat, comp_labels)
    margin_x = 0.015;
    margin_y_bottom = 0.030;
    margin_y_top = 0.006;
    gap_x = 0.060;
    gap_y = 0.036;

    ax_w = (panel_rect(3) - 2 * margin_x - gap_x) / 2;
    ax_h = (panel_rect(4) - margin_y_bottom - margin_y_top - 2 * gap_y) / 3;

    for row = 1:3
        for col = 1:2
            c = (row - 1) * 2 + col;
            pos_x = panel_rect(1) + margin_x + (col - 1) * (ax_w + gap_x);
            pos_y = panel_rect(2) + panel_rect(4) - margin_y_top - row * ax_h - (row - 1) * gap_y;
            ax = axes('Parent', fig, 'Units', 'normalized', 'Position', [pos_x pos_y ax_w ax_h]);
            hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
            set(ax, 'FontName', 'Times New Roman', 'FontSize', 8.5, ...
                'LineWidth', 0.8, 'GridAlpha', 0.15, 'GridColor', [0 0 0]);

            if ~isempty(baseline_mat) && size(baseline_mat, 1) >= c
                plot(ax, 1:size(baseline_mat, 2), baseline_mat(c,:), ...
                    'LineWidth', 1.1, 'Color', [0.10 0.32 0.90]);
            end
            if ~isempty(nd_mat) && size(nd_mat, 1) >= c
                plot(ax, 1:size(nd_mat, 2), nd_mat(c,:), ...
                    'LineWidth', 1.1, 'Color', [0.92 0.12 0.12]);
            end
            ylabel(ax, comp_labels{c}, 'Interpreter', 'tex');
            set_axis_with_panel_label(ax, ternary(row == 3, 'Number of iterations', ''), '');
        end
    end
end


function T = build_metrics_table(results, variant_names)
    n_methods = size(results, 1);
    n_variants = size(results, 2);
    n_rows = n_methods * n_variants;

    method = cell(n_rows, 1);
    variant = cell(n_rows, 1);
    failure_reason = cell(n_rows, 1);
    status = zeros(n_rows, 1);
    steps = zeros(n_rows, 1);
    final_error = zeros(n_rows, 1);
    initial_target_error = zeros(n_rows, 1);
    final_target_error = zeros(n_rows, 1);
    max_error = zeros(n_rows, 1);
    max_target_error = zeros(n_rows, 1);
    mean_jcond = zeros(n_rows, 1);
    max_jcond = zeros(n_rows, 1);
    max_velocity_norm = zeros(n_rows, 1);
    mean_velocity_norm = zeros(n_rows, 1);
    max_increment_norm = zeros(n_rows, 1);
    mean_increment_norm = zeros(n_rows, 1);
    min_boundary = zeros(n_rows, 1);
    fov_violations = zeros(n_rows, 1);
    min_real_boundary = zeros(n_rows, 1);
    real_fov_violations = zeros(n_rows, 1);
    initial_real_target_error = zeros(n_rows, 1);
    final_real_target_error = zeros(n_rows, 1);
    failure_step = zeros(n_rows, 1);
    final_x = zeros(n_rows, 1);
    final_y = zeros(n_rows, 1);
    final_z = zeros(n_rows, 1);

    row = 0;
    for v = 1:n_variants
        for i = 1:n_methods
            row = row + 1;
            m = results(i,v).metrics;
            method{row} = results(i,v).name;
            variant{row} = variant_names{v};
            failure_reason{row} = m.failure_reason;
            status(row) = m.status;
            steps(row) = m.steps;
            final_error(row) = m.final_error;
            initial_target_error(row) = m.initial_target_error;
            final_target_error(row) = m.final_target_error;
            max_error(row) = m.max_error;
            max_target_error(row) = m.max_target_error;
            mean_jcond(row) = m.mean_jcond;
            max_jcond(row) = m.max_jcond;
            max_velocity_norm(row) = m.max_velocity_norm;
            mean_velocity_norm(row) = m.mean_velocity_norm;
            max_increment_norm(row) = m.max_increment_norm;
            mean_increment_norm(row) = m.mean_increment_norm;
            min_boundary(row) = m.min_boundary;
            fov_violations(row) = m.fov_violations;
            min_real_boundary(row) = m.min_real_boundary;
            real_fov_violations(row) = m.real_fov_violations;
            initial_real_target_error(row) = m.initial_real_target_error;
            final_real_target_error(row) = m.final_real_target_error;
            failure_step(row) = m.failure_step;
            final_x(row) = m.final_x;
            final_y(row) = m.final_y;
            final_z(row) = m.final_z;
        end
    end

    T = table(method, variant, status, failure_reason, steps, final_error, ...
        initial_target_error, final_target_error, max_error, max_target_error, ...
        mean_jcond, max_jcond, max_velocity_norm, mean_velocity_norm, ...
        max_increment_norm, mean_increment_norm, min_boundary, ...
        fov_violations, min_real_boundary, real_fov_violations, ...
        initial_real_target_error, final_real_target_error, ...
        failure_step, final_x, final_y, final_z);
end


function write_failure_summary(results, out_dir)
    summary_path = fullfile(out_dir, 'unified_four_method_failure_summary.txt');
    fid = fopen(summary_path, 'w');
    if fid < 0
        warning('Could not write failure summary: %s', summary_path);
        return;
    end

    cleanup_obj = onCleanup(@() fclose(fid));
    fprintf(fid, 'Unified IBVS comparison failure summary (method-feature evaluation)\n');
    fprintf(fid, 'Generated by C_IBVS_PF_NDRG.m\n\n');
    for v = 1:size(results, 2)
        for i = 1:size(results, 1)
            m = results(i,v).metrics;
            if method_uses_virtual_feature(results(i,v).key)
                eval_feature_name = 'virtual f (Cheng control feature)';
            elseif strcmpi(results(i,v).key, 'pf_ndrg')
                eval_feature_name = 'current feature - virtual target s_ref (PF-NDRG)';
            else
                eval_feature_name = 'real_f';
            end
            fprintf(fid, '%s [%s]\n', results(i,v).name, results(i,v).variant);
            fprintf(fid, '  evaluation_feature=%s\n', eval_feature_name);
            fprintf(fid, '  status=%d, reason=%s, steps=%d\n', ...
                m.status, m.failure_reason, m.steps);
            fprintf(fid, '  initial_error=%.6f px, final_error=%.6f px\n', ...
                m.initial_target_error, m.final_target_error);
            fprintf(fid, '  min_boundary=%.6f px, fov_violations=%d, failure_step=%.0f\n', ...
                m.min_boundary, m.fov_violations, m.failure_step);
            if ~isnan(m.min_real_boundary)
                fprintf(fid, '  real_min_boundary=%.6f px, real_fov_violations=%d\n', ...
                    m.min_real_boundary, m.real_fov_violations);
            end
            if m.status < 0 && strcmp(m.failure_reason, 'fov_violation_virtual_feature_left_image')
                fprintf(fid, '  diagnosis=Cheng virtual feature crossed the image boundary before convergence.\n');
            elseif m.status < 0 && strcmp(m.failure_reason, 'fov_violation_feature_left_image')
                fprintf(fid, '  diagnosis=At least one evaluated image feature crossed the image boundary before convergence.\n');
            elseif m.status < 0
                fprintf(fid, '  diagnosis=The controller returned failure before satisfying the error threshold.\n');
            end
            fprintf(fid, '\n');
        end
    end
end


function save_figure(fig, out_dir, base_name)
    clear_figure_titles(fig);
    format_y_axes_for_publication(fig);
    png_path = fullfile(out_dir, [base_name '.png']);
    fig_path = fullfile(out_dir, [base_name '.fig']);
    try
        exportgraphics(fig, png_path, 'Resolution', 300);
    catch
        saveas(fig, png_path);
    end
    try
        savefig(fig, fig_path);
    catch
    end
end


function format_y_axes_for_publication(fig)
    delete(findall(fig, 'Tag', 'manual_y_exponent'));
    axes_handles = findall(fig, 'Type', 'axes');
    for i = 1:numel(axes_handles)
        ax = axes_handles(i);
        if ~is_publication_2d_axis(ax)
            continue;
        end
        try
            if strcmpi(get(ax, 'YScale'), 'log')
                continue;
            end
            y_lim = get(ax, 'YLim');
            y_ticks = get(ax, 'YTick');
        catch
            continue;
        end
        if isempty(y_ticks) || numel(y_lim) ~= 2 || any(~isfinite(y_lim))
            continue;
        end

        max_abs = max(abs(y_lim));
        if max_abs <= 0 || ~isfinite(max_abs)
            continue;
        end
        finite_tick_abs = abs(y_ticks(isfinite(y_ticks) & y_ticks ~= 0));
        has_fractional_scale_tick = ~isempty(finite_tick_abs) && any(finite_tick_abs < 1);
        if max_abs >= 1 && ~has_fractional_scale_tick
            continue;
        end

        exponent = choose_y_tick_exponent(y_ticks, max_abs);
        if exponent >= 0
            continue;
        end
        scaled_ticks = y_ticks ./ (10 ^ exponent);
        labels = cell(size(scaled_ticks));
        for k = 1:numel(scaled_ticks)
            labels{k} = format_scaled_tick_label(scaled_ticks(k), scaled_ticks);
        end

        try
            ax.YAxis.Exponent = 0;
        catch
        end
        set(ax, 'YTickLabel', labels, 'TickLabelInterpreter', 'tex');
        text(ax, 0, 1.025, sprintf('\\times10^{%d}', exponent), ...
            'Units', 'normalized', ...
            'Tag', 'manual_y_exponent', ...
            'FontName', 'Times New Roman', ...
            'FontSize', max(7.5, get(ax, 'FontSize') - 1), ...
            'Interpreter', 'tex', ...
            'HorizontalAlignment', 'left', ...
            'VerticalAlignment', 'bottom', ...
            'Clipping', 'off');
    end
end


function exponent = choose_y_tick_exponent(y_ticks, max_abs)
    finite_ticks = y_ticks(isfinite(y_ticks) & y_ticks ~= 0);
    if isempty(finite_ticks)
        exponent = floor(log10(max_abs));
    else
        exponent = floor(log10(min(abs(finite_ticks))));
    end
    exponent = min(exponent, -1);
end


function tf = is_publication_2d_axis(ax)
    tf = false;
    try
        [az, el] = view(ax);
        tf = abs(az) < 1e-10 && abs(el - 90) < 1e-10;
    catch
    end
end


function label = format_scaled_tick_label(value, all_values)
    max_abs = max(abs(all_values));
    if abs(value) < max(1e-12, max_abs * 1e-12)
        value = 0;
    end
    if value == 0
        label = '0';
    elseif abs(value - round(value)) < 1e-10
        label = sprintf('%d', round(value));
    else
        label = sprintf('%.4f', value);
        label = regexprep(label, '(\.\d*?)0+$', '$1');
        label = regexprep(label, '\.$', '');
    end
    if strcmp(label, '-0')
        label = '0';
    end
end


function clear_figure_titles(fig)
    axes_handles = findall(fig, 'Type', 'axes');
    for i = 1:numel(axes_handles)
        try
            title(axes_handles(i), '');
        catch
        end
    end

    try
        layout_handles = findall(fig, 'Type', 'tiledlayout');
        for i = 1:numel(layout_handles)
            try
                title(layout_handles(i), '');
            catch
            end
        end
    catch
    end

    try
        delete(findall(fig, 'Tag', 'sgtitle'));
        delete(findall(fig, 'Tag', 'suptitle'));
    catch
    end
end


function value = ternary(condition, true_value, false_value)
    if condition
        value = true_value;
    else
        value = false_value;
    end
end
