% IBVS  Image-based visual servoing with the FOV-aware virtual reference (G_s),
% neurodynamic displacement-increment shaping (G_t) and the baseline front-ends
% (OG-VFVR, MPC, APF) used in the comparative simulations of the paper.
% Extends the IBVS class of the Machine Vision Toolbox for MATLAB (P. Corke).


classdef IBVS < VisualServo

    properties
        lambda          % IBVS gain
        lambda_adaptive_enabled
        lambda_min
        lambda_max
        lambda_exp_rate
        lambda_exp_center
        lambda_ramp_steps
        eterm
        uv_p            % previous image coordinates

        depth
        depthest
        vel_p
        raw_vel_p       % [Added] Previous raw velocity command for accel calculation
        theta
        smoothing
        % --- Virtual-FOV smoothing parameters ---
        fov_virtual_smoothing    % [0,1] blending factor of the virtual-FOV smoothing
        fov_virtual_max_step     % maximum per-frame step of the virtual-FOV smoothing (px)
        virtual_uv_prev          % virtual features of the previous frame
            % --- Legacy neurodynamic (ND) image-error model parameters ---
            nd_enabled              % enable the ND displacement-increment shaping G_t
            nd_margin               % inner FOV margin of the legacy ND model (px)
            nd_max_step_pixels      % per-point per-frame step limit of the legacy ND model (px)
            s_ref                   % 2N x 1 reference features (image plane)
            % s_ref 仅用于步长限制的虚拟参考，不再进行二阶速度积分
            % --- Neurodynamic error model parameters (paper-based) ---
            nd_A                    % ND parameter A: passive decay (scalar)
            nd_G                    % ND parameter G: upper bound (scalar)
            nd_D                    % ND parameter D: lower-bound magnitude (scalar)
            nd_E                    % ND state: filtered image error (2N x 1 vector)
    nd_uv_alpha             % smoothing factor for measurement filtering
    nd_uv_filt              % filtered feature measurements (2xN)
    nd_edot_alpha           % smoothing factor for ND derivative
    nd_Edot_prev            % previous filtered neurodynamic derivative
    nd_step_boost_gain      % gain for adaptive step boost (pixels per error)
    nd_step_boost_limit     % max pixel boost for ND step limiting
    nd_first_frame          % flag ensuring first control step starts at zero velocity
    nd_first_frame_trigger  % latched flag for zero-velocity enforcement
    nd_vel_alpha            % low-pass factor for commanded twist
    nd_vel_state            % stored filtered twist command (6x1)
    nd_vel_max_delta        % per-step max change for each twist component
    nd_filter_only          % when true, keep ND smoothing/limits but skip ND dynamics
    legacy_fov_enabled      % Enable Cheng/RVC legacy virtual-FOV correction
    legacy_fov_margin       % Legacy inner FOV warning margin in pixels
    pf_enabled              % Enable PF-NDRG method
        pf_s_ref                % Current state of reference features (2xN)
    pf_attract_gain         % Gain for attraction to target (lambda_ref)
    pf_repulse_gain         % Gain for repulsion from boundaries
    pf_safety_margin        % Distance from boundary to trigger repulsion (pixels)
    pf_initialized          % Flag for initialization
    nd_pos_step_state 

    % --- PF-NDRG: Velocity-level States & Parameters ---
    pf_V_ref                % Reference Velocity State (2xN) for velocity-level smoothing
    pf_damping_gain         % Damping (A) for inertia
    pf_V_max
    paper_apf_enabled       % Enable Corke-Hutchinson partitioned IBVS + APF
    paper_theta_gain        % Gain for optic-axis rotation from line angle
    paper_sigma_gain        % Gain for z translation from sqrt(area)
    paper_repulse_gain      % APF eta at the image boundary
    paper_rho0              % Boundary influence distance in pixels
    paper_mu                % Discrete smoothing coefficient for Tz
    paper_tz_min            % Negative Tz clipping, camera retreat
    paper_tz_max            % Positive Tz clipping, camera approach
    paper_wz_max            % Absolute clipping for omega_z
    paper_tz_state          % Smoothed Tz state
    paper_theta_pair         % Optional fixed point pair [i j] for theta_ij
    paper_delta_max         % APF-IBVS per-component increment limit (m / rad)
    mpc_enabled             % Enable MPC-QP front-end for FOV-constrained IBVS
    mpc_horizon             % MPC prediction horizon
    mpc_q_weight            % MPC feature tracking weight
    mpc_r_weight            % MPC velocity regularization weight
    mpc_slack_weight        % MPC soft FOV constraint slack weight
    mpc_v_max               % MPC translational velocity bound
    mpc_w_max               % MPC angular velocity bound
    mpc_fov_margin          % Image boundary margin for MPC constraints
    mpc_tau_prev            % Previous MPC velocity command used as warm start
    mpc_solve_repeats       % Number of repeated QP solves from the previous solution
    % --- Dual Experiment Results ---
    dual_history_raw        % History from no-ND experiment
    dual_history_nd         % History from ND experiment
    end

    methods

        function ibvs = IBVS(cam, varargin)
            %IBVS.IBVS Create IBVS visual servo object
            %
            % IB = IBVS(camera, options)
            %
            % Options::
            % 'niter',N         Maximum number of iterations
            % 'eterm',E         Terminate when norm of feature error < E
            % 'lambda',L        Control gain, positive definite scalar or matrix
            % 'T0',T            The initial pose
            % 'P',p             The set of world points (3xN)
            % 'targetsize',S    The target points are the corners of an SxS square
            % 'pstar',p         The desired image plane coordinates
            % 'depth',D         Assumed depth of points is D (default true depth
            %                   from simulation is assumed)
            % 'depthest'        Run a simple depth estimator
            % 'fps',F           Number of simulation frames per second (default t)
            % 'verbose'         Print out extra information during simulation
            %
            % Notes::
            % - If 'P' is specified it overrides the default square target.
            %
            % See also VisualServo.

            % invoke superclass constructor
            ibvs = ibvs@VisualServo(cam, varargin{:});
            ibvs.gain_type = 0 ;    % gain type: 0 constant, 1 adaptive, 2 fuzzy
            ibvs.is_train = true;
            % handle arguments
            opt.eterm = 0.5;
            opt.lambda = 0.1;         % control gain
            opt.depth = [];
            opt.depthest = false;
            opt.example = false;

            opt.fuzzylogic = [];


            opt = tb_optparse(opt, ibvs.arglist);

            if opt.example
                % run a canned example
                fprintf('---------------------------------------------------\n');
                fprintf('canned example, image-based IBVS with 4 points\n');
                fprintf('---------------------------------------------------\n');
                ibvs.P = mkgrid(2, 0.5, 'pose', SE3(0,0,3));
                ibvs.pf = bsxfun(@plus, 200*[-1 -1 1 1; -1 1 1 -1], cam.pp');
                ibvs.T0 = SE3(1,1,-3)*SE3.Rz(0.6);
                ibvs.lambda = opt.lambda;
                ibvs.lambda_adaptive_enabled = false;
                ibvs.lambda_min = opt.lambda;
                ibvs.lambda_max = opt.lambda;
                ibvs.lambda_exp_rate = 12.0;
                ibvs.lambda_exp_center = 0.80;
                ibvs.lambda_ramp_steps = 1;
                ibvs.eterm = 0.5;
            else
                % copy options to IBVS object
                ibvs.lambda = opt.lambda;
                ibvs.lambda_adaptive_enabled = false;
                ibvs.lambda_min = opt.lambda;
                ibvs.lambda_max = opt.lambda;
                ibvs.lambda_exp_rate = 12.0;
                ibvs.lambda_exp_center = 0.80;
                ibvs.lambda_ramp_steps = 1;
                ibvs.eterm = opt.eterm;
                ibvs.theta = 0;
                ibvs.smoothing = 0.80;
                ibvs.depth = opt.depth;
                ibvs.depthest = opt.depthest;
                % Virtual-FOV smoothing defaults
                ibvs.fov_virtual_smoothing = 0.5;   % 0(关闭)~1(全开)
                ibvs.fov_virtual_max_step  = 30;    % 每帧虚拟FOV平滑的最大步长
                ibvs.virtual_uv_prev = [];
                    % Neurodynamic error model (paper-based)
                     ibvs.nd_enabled = true;
                    % ibvs.nd_enabled = false;
                    ibvs.nd_margin = 100;           % FOV内缩边界（像素）针对1024x1024
                    ibvs.nd_max_step_pixels = 18;   % 每点每帧最大像素步长，33fps更稳
                    ibvs.s_ref = [];
                    % Paper-based ND defaults (tuned)
                    ibvs.nd_A = 10;              % 针对33fps的稳健耗散
                    ibvs.nd_G = 10;               % E正向上界（像素），1024分辨率建议
                    ibvs.nd_D = 10;               % E负向下界幅值（像素）
                    ibvs.nd_E = [];
                    ibvs.nd_uv_alpha = 0.35;      % measurement低通
                    ibvs.nd_uv_filt = [];
                    ibvs.nd_edot_alpha = 1;    % derivative低通
                    ibvs.nd_Edot_prev = [];
                    ibvs.nd_step_boost_gain = 0.055;  % adaptive step gain (pixels per pixel error)
                    ibvs.nd_step_boost_limit = 18;    % max extra pixels per frame
                    ibvs.nd_first_frame = true;
                    ibvs.nd_first_frame_trigger = false;
                    ibvs.nd_vel_alpha = 0.32;
                    ibvs.nd_vel_state = [];
                    ibvs.nd_vel_max_delta = [0.012 0.012 0.012 0.0035 0.0035 0.0035]';
                    % ibvs.nd_filter_only = false;   % allow keeping filters/limits without ND dynamics
                    ibvs.nd_filter_only = true;  
                    ibvs.legacy_fov_enabled = false;
                    ibvs.legacy_fov_margin = 100;
                    % 默认帧率（若上层未设置），与33fps匹配
                    try
                        if ~isfield(ibvs, 'fps') || isempty(ibvs.fps) || ~isnumeric(ibvs.fps) || ibvs.fps<=0
                            ibvs.fps = 33;
                        end
                    catch
                        ibvs.fps = 33;
                    end
                      % Paper-based neurodynamic error model default parameters
            ibvs.nd_A = 10.0;               % error damping A
            ibvs.nd_G = 10.0;               % upper bound G
            ibvs.nd_D = 10.0;               % lower bound magnitude D
            ibvs.nd_E = [];                % dynamic error state (initialized later)
            ibvs.nd_uv_alpha = 0.35;
            ibvs.nd_uv_filt = [];
            ibvs.nd_edot_alpha = 1;
            ibvs.nd_Edot_prev = [];
            ibvs.nd_step_boost_gain = 0.055;
            ibvs.nd_step_boost_limit = 18;
            ibvs.nd_first_frame = true;
            ibvs.nd_first_frame_trigger = false;
            ibvs.nd_vel_alpha = 0.32;
            ibvs.nd_vel_state = [];
            ibvs.nd_vel_max_delta = [0.012 0.012 0.012 0.0035 0.0035 0.0035]';
            ibvs.legacy_fov_enabled = false;
            ibvs.legacy_fov_margin = 100;
            ibvs.paper_apf_enabled = false;
            ibvs.paper_theta_gain = 0.35;
            ibvs.paper_sigma_gain = 0.01;
            ibvs.paper_repulse_gain = 5e6;
            ibvs.paper_rho0 = 50;
            ibvs.paper_mu = 0.8;
            ibvs.paper_tz_min = -1.0;
            ibvs.paper_tz_max = 0.35;
            ibvs.paper_wz_max = 0.45;
            ibvs.paper_tz_state = 0;
            ibvs.paper_theta_pair = [];
            ibvs.paper_delta_max = 0.05;
            ibvs.mpc_enabled = false;
            ibvs.mpc_horizon = 6;
            ibvs.mpc_q_weight = 10.0;
            ibvs.mpc_r_weight = 1.0;
            ibvs.mpc_slack_weight = 1e5;
            ibvs.mpc_v_max = 0.5;
            ibvs.mpc_w_max = 0.5;
            ibvs.mpc_fov_margin = 0;
            ibvs.mpc_tau_prev = zeros(6,1);
            ibvs.mpc_solve_repeats = 1;

            end

            % Reset the virtual-FOV state when the FOV correction is inactive
            try
                if ibvs.is_FOV == 0
                    ibvs.virtual_uv_prev = [];
                end
            catch
            end


            ibvs.camera.T=SE3.convert(ibvs.T0);
            ibvs.Tcam=SE3.convert(ibvs.T0);
            if ~isempty(ibvs.axis)
                figure
            axis(ibvs.axis);
            end
            plot_sphere(ibvs.P, 0.05, 'r')
            lighting gouraud
            light
            %cam2 = showcamera(T0);
            ibvs.camera.plot_camera(ibvs.P, 'label');
            ibvs.type = 'point';
        end

        function init(vs)

            if ~isempty(vs.pf)
                % final pose is specified in terms of image coords
                vs.uv_star = vs.pf;
            else
                if ~isempty(vs.Tf)
                    vs.Tf = transl(0, 0, 1);
                    warning('setting Tf to default');
                end
                % final pose is specified in terms of a camera-target pose
                %   convert to image coords
                vs.uv_star = vs.camera.project(vs.P, 'Tcam', inv(vs.Tf));
            end

            % initialize the vservo variables
            vs.camera.T = vs.T0;    % set camera back to its initial pose
            vs.Tcam = vs.T0;                % initial camera/robot pose
            % reset neurodynamic filters for a fresh run
            vs.nd_E = [];
            vs.s_ref = [];
            vs.nd_uv_filt = [];
            vs.nd_Edot_prev = [];
            vs.nd_first_frame = true;
            vs.nd_first_frame_trigger = false;
            vs.nd_vel_state = [];
            vs.paper_tz_state = 0;
            vs.mpc_tau_prev = zeros(6,1);

            vs.vel_p = [];
            vs.raw_vel_p = []; % [Added]
            vs.uv_p = [];
            vs.history = [];
        end

        function status = step(vs)

            status = 0;
            Zest = [];

            % compute the view
            uv = vs.camera.plot(vs.P);
            real_uv = uv;
            if vs.is_first_cal == 1
                vs.last_uv = uv;
            end

            % Keep the measured features; virtual features are produced by the selected front-end
            virtual_uv_base = uv;
            virtual_uv = uv;
            vs.s_ref = [];
            raw_legacy_enabled = logical(~isempty(vs.legacy_fov_enabled) && vs.legacy_fov_enabled);
            raw_pf_enabled = logical(~isempty(vs.pf_enabled) && vs.pf_enabled);
            raw_paper_enabled = logical(~isempty(vs.paper_apf_enabled) && vs.paper_apf_enabled);
            raw_mpc_enabled = logical(~isempty(vs.mpc_enabled) && vs.mpc_enabled);
            if nnz([raw_legacy_enabled, raw_pf_enabled, raw_paper_enabled, raw_mpc_enabled]) > 1
                error('IBVS:ConflictingMethodFlags', ...
                    ['Conflicting method flags are enabled simultaneously. ' ...
                     'legacy_fov=%d, pf_ndrg=%d, paper_apf=%d, mpc=%d'], ...
                    raw_legacy_enabled, raw_pf_enabled, raw_paper_enabled, raw_mpc_enabled);
            end
            use_paper_frontend = logical(~isempty(vs.paper_apf_enabled) && vs.paper_apf_enabled);
            use_mpc_frontend = logical(~isempty(vs.mpc_enabled) && vs.mpc_enabled);
            use_pf_frontend = logical(~isempty(vs.pf_enabled) && vs.pf_enabled) && ...
                ~use_paper_frontend && ~use_mpc_frontend;
            use_legacy_fov = logical(~isempty(vs.legacy_fov_enabled) && vs.legacy_fov_enabled) && ...
                ~use_paper_frontend && ~use_mpc_frontend && ~use_pf_frontend;
            legacy_fov_info = [];
            pf_e_for_history = [];
            pf_s_ref_for_history = [];
            if use_legacy_fov
                [uv, legacy_fov_info] = vs.legacy_fov_virtual_uv(uv);
                virtual_uv = uv;
            end
            if use_pf_frontend
                % === PF-NDRG: Potential-Field Augmented Neurodynamic Reference Governor ===
                if isempty(vs.pf_s_ref) || vs.is_first_cal == 1
                    vs.pf_s_ref = uv;
                    vs.pf_V_ref = zeros(size(uv)); % Initialize velocity state
                    vs.pf_initialized = true;
                    % Default parameters
                    if isempty(vs.pf_attract_gain), vs.pf_attract_gain = 2.0; end
                    if isempty(vs.pf_repulse_gain), vs.pf_repulse_gain = 10000.0; end
                    if isempty(vs.pf_safety_margin), vs.pf_safety_margin = 50; end

                    % Initialize ND Parameters
                    if isempty(vs.pf_damping_gain), vs.pf_damping_gain = 5.0; end % Inertia (A)
                    if isempty(vs.pf_V_max), vs.pf_V_max = 500.0; end             % Saturation (B/D)
                end

                % [Robustness]: Use actual stochastic cycle time if available
                if isprop(vs, 'Task_process_clcle') && ~isempty(vs.Task_process_clcle)
                    dt = vs.Task_process_clcle * 0.001; % ms -> s
                else
                    dt = 1/vs.fps;
                end

                s_star = vs.uv_star;
                pf_s_ref_for_history = vs.pf_s_ref;
                pf_e_for_history = uv(:) - pf_s_ref_for_history(:);

                % 1. Attraction Force: s_ref -> s_star
                F_att = vs.pf_attract_gain * (s_star - vs.pf_s_ref);

                % 2. Repulsion Force: Triggered by CURRENT REAL FEATURE (uv) proximity to boundary
                % [Critical Logic]: As stated in the methodology, repulsion must be triggered by the 
                % ACTUAL danger (uv), not the virtual plan (s_ref).
                W = vs.camera.npix(1);
                H = vs.camera.npix(2);
                margin = vs.pf_safety_margin;
                F_rep = zeros(size(uv));

                for i = 1:size(uv, 2)
                    u_c = uv(1,i); v_c = uv(2,i); % CHANGED: Use
                    % Left boundary
                    if u_c < margin
                        dist = max(u_c, 1e-3);
                        F_rep(1,i) = F_rep(1,i) + vs.pf_repulse_gain * (1/dist - 1/margin)^2;
                    end
                    % Right boundary
                    if u_c > W - margin
                        dist = max(W - u_c, 1e-3);
                        F_rep(1,i) = F_rep(1,i) - vs.pf_repulse_gain * (1/dist - 1/margin)^2;
                    end
                    % Top boundary
                    if v_c < margin
                        dist = max(v_c, 1e-3);
                        F_rep(2,i) = F_rep(2,i) + vs.pf_repulse_gain * (1/dist - 1/margin)^2;
                    end
                    % Bottom boundary
                    if v_c > H - margin
                        dist = max(H - v_c, 1e-3);
                        F_rep(2,i) = F_rep(2,i) - vs.pf_repulse_gain * (1/dist - 1/margin)^2;
                    end
                end

                % 3. Apply Shunting Model Dynamics (Velocity-level Smoothing)
                % Incorporates integral inertia to smooth out jitter from stochastic dt

                F_total = F_att + F_rep;
                % Eq. (16): forward-Euler update, then projection onto Omega_eps
                pf_eps = 1;   % interior margin epsilon (px), 0 < eps < mu
                vs.pf_s_ref = vs.pf_s_ref + F_total * dt;
                vs.pf_s_ref(1,:) = min(max(vs.pf_s_ref(1,:), pf_eps), W - pf_eps);
                vs.pf_s_ref(2,:) = min(max(vs.pf_s_ref(2,:), pf_eps), H - pf_eps);
                vs.s_ref = vs.pf_s_ref(:);
                virtual_uv = vs.pf_s_ref;

            end

            % optionally estimate depth
            if vs.depthest
                % run the depth estimator
                [Zest,Ztrue] = vs.depth_estimator(uv);
                if vs.verbose
                    fprintf('Z: est=%f, true=%f\n', Zest, Ztrue)
                end   
                if isempty(Zest) == 1
                    vs.desired_depth = vs.depth;
                end
                vs.depth = Zest;
                hist.Ztrue = Ztrue(:);
                hist.Zest = Zest(:);
            end



   % compute image plane error as a column
            if use_paper_frontend
                e = uv(:) - vs.uv_star(:);
            elseif use_pf_frontend
                e = uv(:) - vs.pf_s_ref(:);
            else
                e = uv(:) - vs.uv_star(:);
            end
            e = e(:);

            % Interaction matrix: exact desired depth, fuzzy-blended estimated depth,
            % or a constant assumed depth
            if isempty(vs.depth)
                % exact depth from simulation (not possible in practice)
                J = vs.camera.visjac_p(uv, vs.desired_depth);
                vs.error_bate_last = e;
            elseif ~isempty(Zest)
                Jc = vs.camera.visjac_p(uv, Zest);
                Jd = vs.camera.visjac_p(uv, vs.desired_depth);
                % fuzzy blending weight beta between current and desired Jacobians
                error_beta = sqrt(sum(e.^2));
                error_beta_uni = error_beta / 1448;
                cycle = 1/vs.fps * 1000;
                error_beta_rate = (error_beta - sqrt(sum(vs.error_bate_last.^2)))/cycle;
                error_beta_rate_uni = error_beta_rate / 48.27;
                vs.beta_jacobian = evalfis(vs.fis_jacobian,[error_beta_uni error_beta_rate_uni]);
%                 vs.beta_jacobian = 0.5;
%                 disp(vs.beta_jacobian);
                vs.error_bate_last = e;
                % blended interaction matrix
                J = vs.beta_jacobian * Jc + (1 - vs.beta_jacobian) * Jd;
            else
                J = vs.camera.visjac_p(uv, vs.depth);
            end

            % gain_type==3（强化学习）未使用，直接跳过

            % compute the velocity of camera in camera frame
            try
                if ~use_mpc_frontend
                %------- gain_type 0: constant gain, record per-axis maxima of s
                if vs.gain_type == 0
                    s = pinv(J) * e;
                    if abs(vs.max_s1) < abs(s(1))
                        vs.max_s1 = s(1);
                    end
                    if abs(vs.max_s2) < abs(s(2))
                        vs.max_s2 = s(2);
                    end
                    if abs(vs.max_s3) < abs(s(3))
                        vs.max_s3 = s(3);
                    end
                    if abs(vs.max_s4) < abs(s(4))
                        vs.max_s4 = s(4);
                    end
                    if abs(vs.max_s5) < abs(s(5))
                        vs.max_s5 = s(5);
                    end
                    if abs(vs.max_s6) < abs(s(6))
                        vs.max_s6 = s(6);
                    end
                end
                %------- gain_type 1: adaptive gain
                if vs.gain_type ==1
                    TK=pinv(J) * e;
                    input = norm(TK , "inf");
                    gian_at_zero = 1.5; gain_at_infinity = 0.4; slope_at_zero = 30;
                    coeff_a = gian_at_zero - gain_at_infinity;
                    coeff_b = slope_at_zero / coeff_a;
                    coeff_c = gain_at_infinity;
                    vs.lambda = coeff_a * exp(-coeff_b * input) + coeff_c;
                end
                %-------- gain_type 2: fuzzy-logic gain
                if vs.gain_type == 2
                    % normalised error and error rate as fuzzy inputs
                    error = sqrt(sum(e.^2));
                    error_uni = error / 1448;
                    cycle = 1/vs.fps * 1000;
                    error_rate = (error - sqrt(sum(vs.e_last.^2)))/cycle;
                    error_rate_uni = error_rate / 48.27;
                    vs.lambda = evalfis(vs.fis,[error_uni error_rate_uni]);
                    vs.e_last = e;
                end
                %--------- end of gain selection
                % 使用 gain_type 0/1/2 的标准控制律
                if ~isempty(vs.lambda_adaptive_enabled) && vs.lambda_adaptive_enabled
                    k_lambda = numel(vs.history) + 1;
                    ramp_steps = max(1, vs.lambda_ramp_steps);
                    xi = min(1.0, max(0.0, (k_lambda - 1) / ramp_steps));
                    beta = vs.lambda_exp_rate;
                    center = vs.lambda_exp_center;
                    if isempty(beta), beta = 12.0; end
                    if isempty(center), center = 0.80; end
                    beta = max(beta, 1e-6);
                    center = min(0.99, max(0.01, center));
                    sig = 1 / (1 + exp(-beta * (xi - center)));
                    sig0 = 1 / (1 + exp(beta * center));
                    sig1 = 1 / (1 + exp(-beta * (1 - center)));
                    ramp = (sig - sig0) / max(sig1 - sig0, eps);
                    ramp = min(1.0, max(0.0, ramp));
                    vs.lambda = vs.lambda_min + (vs.lambda_max - vs.lambda_min) * ramp;
                end
                if use_paper_frontend
                    [Tz_paper, wz_paper, paper_info] = vs.partitioned_apf_z_control(uv, vs.uv_star);
                    fdot = -vs.lambda * e;
                    Jxy = J(:, [1 2 4 5]);
                    Jz = J(:, [3 6]);
                    rxy = pinv(Jxy) * (fdot - Jz * [Tz_paper; wz_paper]);
                    v = [rxy(1); rxy(2); Tz_paper; rxy(3); rxy(4); wz_paper];
                else
                    v_cmd = -vs.lambda * pinv(J) * e;
                    v = v_cmd;
                end
                else
                    if isprop(vs, 'Task_process_clcle') && ~isempty(vs.Task_process_clcle)
                        dt_mpc = vs.Task_process_clcle * 0.001;
                    else
                        dt_mpc = 1 / max(vs.fps, 1);
                    end
                    [v_cmd, s_ref_pred, exitflag] = vs.mpcqp_controller_pixel(uv, vs.uv_star, J, dt_mpc);
                    if exitflag > 0 && ~isempty(v_cmd)
                        vs.s_ref = s_ref_pred;
                        vs.mpc_tau_prev = v_cmd;
                        virtual_uv = reshape(s_ref_pred, size(uv));
                        v = v_cmd;
                    else
                        status = -1;
                        return;
                    end
                end

            catch
                status = -1;
                return
            end

            if vs.verbose
                fprintf('v: %.3f %.3f %.3f %.3f %.3f %.3f\n', v);
            end

            %====== update the camera pose with the executed displacement increment
            % v_delta=v*1/vs.fps;
            % raw_v_delta=v*1/vs.fps;
            % Eq. (20): raw displacement increment uses the actual sampling interval dt_k
            if isprop(vs, 'Task_process_clcle') && ~isempty(vs.Task_process_clcle)
                dt_step = vs.Task_process_clcle * 0.001;
            else
                dt_step = 1 / max(vs.fps, 1);
            end
            raw_v_delta = v * dt_step;
            if use_paper_frontend && ~isempty(vs.paper_delta_max)
                % APF-IBVS baseline: each increment component limited to paper_delta_max
                raw_v_delta = min(max(raw_v_delta, -vs.paper_delta_max), vs.paper_delta_max);
            end

  if vs.nd_enabled
                 % Initialize filtering state
                % NOTE: State must be stored in SCALED Domain to maintain ND dynamics
                if isempty(vs.nd_pos_step_state) || numel(vs.nd_pos_step_state) ~= numel(raw_v_delta)
                    vs.nd_pos_step_state = zeros(size(raw_v_delta));
                end

                % [Scale-Filter-Descale Architecture]
                % 1. Scale Up: Transform tiny steps (~1e-4) to unit domain (~1.0).
                % 2. Smooth: Apply ND filter with High Inertia (Low A).
                % 3. Unity Gain Logic: Ensure B*K/A approx 1 to maintain trajectory scale.

                % "High Smoothness" Configuration:
                K_gain = 8.0;       % Original good-ND scaling; keeps B*K/A near 1.

                % --- Step 1: Scale Up Input ---


                % ND Parameters 
                A_step = 0.4;       % Original good-ND decay for stronger smoothing.
                B_step = 0.05;      % Saturation limit (Physical safety bound)
                D_step = 0.05;
                % Inputs (Rectified)
                  I_plus  = max(0, raw_v_delta * K_gain);
                I_minus = max(0, -raw_v_delta * K_gain);

                % --- Step 2: ND Smoothing (In Scaled Domain) ---
                % We assume vs.nd_pos_step_state stores the PREVIOUS SCALED STATE.
                % If the previous run stored 'small' state, this will auto-correct in a few steps.
                Step_curr = vs.nd_pos_step_state; 

                dStep = -A_step * Step_curr + (B_step - Step_curr) .* I_plus - (D_step + Step_curr) .* I_minus;
                Step_next = Step_curr + dStep;

                % Safety Clamp (In Scaled Domain)
                Step_next = min(B_step, max(-D_step, Step_next));

                % Update State (Keep it SCALED for next iteration!)
                vs.nd_pos_step_state = Step_next;

                % --- Step 3: Scale Down Output ---
                v_delta = Step_next;

            else
                 v_delta = raw_v_delta;
            end





            %======================= apply the differential motion on SE(3)
            Td = SE3(trnorm(delta2tr(v_delta)));    % differential motion
            %Td = expm( skewa(v) );
            %Td = SE3( delta2tr(v) );
            vs.Tcam = vs.Tcam .* Td;       % apply it to current pose
            %vs.Tcam = trnorm(vs.Tcam);    
            % update the camera pose
            vs.camera.T = vs.Tcam;
            %======================= end of pose update




                  hist.f = uv(:);
            hist.real_f = real_uv(:);
            if ~isempty(legacy_fov_info)
                hist.legacy_fov_info = legacy_fov_info;
            end
            hist.raw_vel = v;        % [Added] Raw Command Velocity

            % Save ACTUAL time step used for this frame (for plotting jitter)
            if isprop(vs, 'Task_process_clcle') && ~isempty(vs.Task_process_clcle)
                 hist.dt = vs.Task_process_clcle * 0.001;
            else
                 hist.dt = 1/vs.fps;
            end

            vel = tr2delta(Td);
            hist.vel = vel;
            if use_pf_frontend && ~isempty(pf_e_for_history)
                hist.e = pf_e_for_history;
                hist.en = norm(pf_e_for_history);
                hist.s_ref_error = pf_s_ref_for_history;
            else
                hist.e = e;
                hist.en = norm(e);
                hist.s_ref_error = [];
            end
            hist.jcond = cond(J);
            if exist('paper_info','var') && ~isempty(paper_info)
                hist.paper_info = paper_info;
            end
            hist.Tcam = vs.Tcam;
            if exist('E_new','var')
                hist.nd_E = E_new;
                hist.nd_en = norm(E_new);
            else
                hist.nd_E = [];
                hist.nd_en = NaN;
            end
            hist.raw_delta = raw_v_delta;
            hist.executed_delta = v_delta;
            hist.lambda = vs.lambda;
            if exist('s_new','var')
                hist.s_ref = s_new;
            elseif ~isempty(vs.s_ref)
                hist.s_ref = reshape(vs.s_ref, size(uv));
            elseif use_pf_frontend
                hist.s_ref = vs.pf_s_ref;
            else
                hist.s_ref = [];
            end

            % 计算加速度（速度变化率）
            % vel 和 vel_p 是位移增量(m/rad)，先除以各自的Δt得到速度，再差分得到加速度
            dt_curr = hist.dt;
            if ~isempty(vs.vel_p) && isfield(vs.history(end), 'dt') && ~isempty(vs.history(end).dt)
                dt_prev = vs.history(end).dt;
                actual_vel_curr = vel / dt_curr;     % 当前速度 (m/s)
                actual_vel_prev = vs.vel_p / dt_prev; % 上一步速度 (m/s)
                accel = (actual_vel_curr - actual_vel_prev) / dt_curr;
                hist.accel = accel;
                hist.accel_norm = norm(accel);
            elseif ~isempty(vs.vel_p)
                accel = (vel - vs.vel_p) / dt_curr;
                hist.accel = accel;
                hist.accel_norm = norm(accel);
            else
                hist.accel = zeros(6,1);
                hist.accel_norm = 0;
            end

            % [Added] Calculate Raw Acceleration
             if ~isempty(vs.raw_vel_p)
                raw_accel = (v - vs.raw_vel_p) / dt_curr;
                hist.raw_accel = raw_accel;
            else
                hist.raw_accel = zeros(6,1);
            end

            % 计算位置轨迹曲率（基于相机位置）
            if length(vs.history) >= 2
                pos_curr = vs.Tcam.t;
                pos_prev = vs.history(end).Tcam.t;
                pos_prev2 = vs.history(end-1).Tcam.t;

                v1 = pos_curr - pos_prev;
                v2 = pos_prev - pos_prev2;

                % 曲率计算：|v1 × v2| / |v1|³
                cross_prod = cross(v1, v2);
                v1_norm = norm(v1);
                if v1_norm > 1e-8
                    hist.curvature = norm(cross_prod) / (v1_norm^3);
                else
                    hist.curvature = 0;
                end
            else
                hist.curvature = 0;
            end

            % 计算图像误差平滑度指标
            if ~isempty(vs.history)
                e_prev = vs.history(end).e;
                e_diff = e - e_prev;
                hist.error_smoothness = norm(e_diff);

                % 如果有ND误差，也计算ND误差的平滑度
                if exist('E_new','var') && isfield(vs.history(end), 'nd_E') && ~isempty(vs.history(end).nd_E)
                    nd_e_prev = vs.history(end).nd_E;
                    nd_e_diff = E_new - nd_e_prev;
                    hist.nd_error_smoothness = norm(nd_e_diff);
                else
                    hist.nd_error_smoothness = NaN;
                end
            else
                hist.error_smoothness = 0;
                hist.nd_error_smoothness = NaN;
            end

            vs.history = [vs.history hist];

            vs.vel_p = vel;
            vs.raw_vel_p = v; % [Added]
            vs.uv_p = uv;

            % (FOV 处理相关注释已清理)
            vs.is_first_cal = 0;

            if norm(e) < vs.eterm
                status = 1;
                return
            end
            % 图像边界检查（注释已清理）
            imsz = vs.camera.npix(:);
            for j = 1 : size(uv,2)
                if uv(1,j) <= 0 || uv(1,j) >= imsz(1) || uv(2,j) <= 0 || uv(2,j) >= imsz(2)
                    status = -1;
                end
            end   

        end

        function plot_nd(vs)
            % plot_nd Plot per-component errors over time (raw e vs ND-smoothed E)
            if isempty(vs.history)
                warning('No history to plot. Run ibvs.run() first.');
                return;
            end
            n = numel(vs.history);

            % collect per-step vectors
            e_mat = [];
            nd_mat = [];
            for k = 1:n
                if isfield(vs.history(k),'e') && ~isempty(vs.history(k).e)
                    e_mat(:,k) = vs.history(k).e(:);
                end
                if isfield(vs.history(k),'nd_E') && ~isempty(vs.history(k).nd_E)
                    nd_mat(:,k) = vs.history(k).nd_E(:);
                end
            end

            % build nice legends: u1,v1,u2,v2...
            ncomp = size(e_mat,1);
            npts = ncomp/2;
            e_labels = cell(1, ncomp*1);
            nd_labels = cell(1, ncomp*1);
            for i=1:npts
                e_labels{2*i-1} = sprintf('e_u%d', i);
                e_labels{2*i}   = sprintf('e_v%d', i);
                nd_labels{2*i-1} = sprintf('E_ND_u%d', i);
                nd_labels{2*i}   = sprintf('E_ND_v%d', i);
            end

            t = 1:n;
            h = figure('Name','ND per-component errors','NumberTitle','off'); clf; hold on;
            if ~isempty(e_mat)
                ph = plot(t, e_mat', '-', 'LineWidth', 1);
            end
            if ~isempty(nd_mat)
                ph2 = plot(t, nd_mat', '--', 'LineWidth', 1.25);
            end

            % build legend depending on presence
            legend_entries = {};
            if ~isempty(e_mat)
                legend_entries = [legend_entries e_labels];
            end
            if ~isempty(nd_mat)
                legend_entries = [legend_entries nd_labels];
            end
            if ~isempty(legend_entries)
                legend(legend_entries, 'Interpreter','none', 'Location','bestoutside');
            else
                title('No ND history recorded');
            end
            xlabel('Step'); ylabel('Error (pixels)');
            grid on; hold off;
            title('Per-component image error vs ND-smoothed error');
        end

        % function plot_nd_step_response(vs, e_vals, nsteps)
        %     % plot ND step response for constant scalar errors
        %     if nargin < 2 || isempty(e_vals)
        %         e_vals = [1 2 3];
        %     end
        %     if nargin < 3 || isempty(nsteps)
        %         nsteps = 200;
        %     end
        % 
        %     % parameters and timestep
        %     if ~isempty(vs.fps) && isnumeric(vs.fps) && vs.fps > 0
        %         dt = 1 / vs.fps;
        %     else
        %         dt = 0.03;
        %     end
        %     A = vs.nd_A; if isempty(A), A = 1.0; end
        %     G = vs.nd_G; if isempty(G), G = 1.0; end
        %     D = vs.nd_D; if isempty(D), D = 1.0; end
        % 
        %     t = (0:nsteps) * dt;
        %     E_hist = zeros(numel(e_vals), numel(t));
        % 
        %     for idx = 1:numel(e_vals)
        %         e_const = e_vals(idx);
        %         E = e_const; % match controller init (E starts from raw error)
        %         E_hist(idx,1) = E;
        %         for k = 2:numel(t)
        %             if e_const >= 0
        %                 f_e = e_const; g_e = 0;
        %             else
        %                 f_e = 0; g_e = -e_const;
        %             end
        %             E_dot = -A * E + (G - E) * f_e - (D + E) * g_e;
        %             E = E + dt * E_dot;
        %             E = min(G - 1e-6, max(-D + 1e-6, E));
        %             E_hist(idx,k) = E;
        %         end
        %     end
        % 
        %     figure;
        %     plot(t, E_hist');
        %     grid on;
        %     xlabel('Time (s)'); ylabel('E_{ND} response (pixels)');
        %     legend(arrayfun(@(x) sprintf('e=%.2f', x), e_vals, 'UniformOutput', false));
        %     title('ND step response to constant scalar errors');
        % end
                function plot_nd_step_response(vs, e_vals, nsteps)
            % plot ND step response for constant scalar errors
  if nargin < 2 || isempty(e_vals), e_vals = [-8 -4 -2 -1 1 2 4 8]; end
            if nargin < 3 || isempty(nsteps), nsteps = 300; end   % 200*0.01 ≈ 2s
            if nargin < 4 || isempty(A), A = 1.0; end
            if nargin < 5 || isempty(G), G = 8.0; end
            if nargin < 6 || isempty(D), D = 8.0; end
            if nargin < 7 || isempty(dt), dt = 0.03; end

            t = (0:nsteps) * dt;
            E_hist = zeros(numel(e_vals), numel(t));
            for idx = 1:numel(e_vals)
                e_const = e_vals(idx);
                E = 0; E_hist(idx,1) = E;  % E(0)=0 per paper
                for k = 2:numel(t)
                    if e_const >= 0, f_e = e_const; g_e = 0; else, f_e = 0; g_e = -e_const; end
                    E_dot = -A * E + (G - E) * f_e - (D + E) * g_e;
                    E = E + dt * E_dot;
                    E = min(G - 1e-6, max(-D + 1e-6, E));
                    E_hist(idx,k) = E;
                end
            end
            figure; plot(t, E_hist', 'LineWidth', 1.2); grid on;
            xlabel('Time (s)', 'FontSize', 11);
            ylabel('ND output $E(t)$', 'Interpreter', 'latex', 'FontSize', 11);
            legend(arrayfun(@(x) sprintf('$e=%.0f$', x), e_vals, 'UniformOutput', false), ...
                   'Interpreter', 'latex', 'Location','best', 'FontSize', 9);
        end

        function plot_nd_online_demo(vs, T)
            % plot_nd_online_demo Simulate online ND response to a time-varying raw error
            % Uses current vs.nd_A/G/D and fps defaults, without altering controller state.
            if nargin < 2 || isempty(T), T = 12; end

            % parameters from the controller (no changes to stored values)
            A = vs.nd_A; if isempty(A), A = 2.0; end
            G = vs.nd_G; if isempty(G), G = 1.5; end
            D = vs.nd_D; if isempty(D), D = 1.5; end
            if ~isempty(vs.fps) && isnumeric(vs.fps) && vs.fps > 0
                dt = 1 / vs.fps;
            else
                dt = 0.03;
            end

            nsteps = round(T / dt);
            t = (0:nsteps) * dt;
            E_hist = zeros(1, nsteps + 1);
            e_hist = zeros(1, nsteps + 1);

            % Initialize state per paper: E(0) = 0
            E = 0;
            E_hist(1) = E;

            % piecewise raw error profile (scalar demo) with small noise
            for k = 1:nsteps+1
                tk = t(k);
                if tk < 3
                    e_raw = 10;
                elseif tk < 6
                    e_raw = -8;
                elseif tk < 9
                    e_raw = 4 ;
                    % * sin(2 * pi * 0.5 * (tk - 6))
                else
                    e_raw = 0;
                end
                e_raw = e_raw + 2 * randn();  % small noise ripple
                e_hist(k) = e_raw;

                % ND dynamics
                f_e = max(0, 10*e_raw);
                g_e = max(0, -e_raw);
                E_dot = -A * E + (G - E) * f_e - (D + E) * g_e;
                E = E + dt * E_dot;
                E = min(G - 1e-6, max(-D + 1e-6, E));
                E_hist(k) = E;
            end

            % plot
            figure;
            subplot(2,1,1);
            plot(t, e_hist, 'b', 'LineWidth', 1.2); grid on;
            ylabel('Raw error $e(t)$', 'Interpreter', 'latex', 'FontSize', 11);

            subplot(2,1,2);
            plot(t, E_hist, 'r', 'LineWidth', 1.2); grid on;
            ylabel('ND state $E(t)$', 'Interpreter', 'latex', 'FontSize', 11);
            xlabel('Time (s)', 'FontSize', 11);
            legend({'$E$ (clamped)'}, 'Interpreter', 'latex', 'Location','best', 'FontSize', 9);
        end

        function plot_acceleration_analysis(vs)
            % 速度变化率（加速度）分析
            if isempty(vs.history)
                warning('No history to plot. Run ibvs.run() first.');
                return;
            end

            % 提取加速度数据
            accel_data = [vs.history.accel]';  % Nx6矩阵
            accel_norms = [vs.history.accel_norm];
            n = length(vs.history);
            t = 1:n;

            % 创建子图
            h = figure('Name','Acceleration Analysis','NumberTitle','off');

            % 子图1：各轴加速度分量
            subplot(2,2,1);
            plot(t, accel_data(:,1:3), '-', 'LineWidth', 1.2);
            hold on;
            plot(t, accel_data(:,4:6), '--', 'LineWidth', 1.2);
            grid on;
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
            ylabel('Acceleration', 'FontSize', 11);
            legend('$a_x$','$a_y$','$a_z$','$\dot{\omega}_x$','$\dot{\omega}_y$','$\dot{\omega}_z$', ...
                   'Interpreter', 'latex', 'FontSize', 8, 'Location', 'best');

            % 子图2：加速度范数
            subplot(2,2,2);
            plot(t, accel_norms, 'r-', 'LineWidth', 1.5);
            grid on;
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
            ylabel('$\|\mathbf{a}\|$', 'Interpreter', 'latex', 'FontSize', 11);

            % 子图3：加速度统计
            subplot(2,2,3);
            accel_stats = [mean(accel_norms), std(accel_norms), max(accel_norms)];
            bar(accel_stats);
            set(gca, 'XTickLabel', {'Mean','Std','Max'});
            ylabel('Acceleration', 'FontSize', 11);
            grid on;

            % 子图4：平滑度指标（加速度变化率）
            subplot(2,2,4);
            if n > 2
                accel_smoothness = diff(accel_norms);
                plot(2:n, abs(accel_smoothness), 'g-', 'LineWidth', 1.2);
                grid on;
                xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
                ylabel('$|\Delta a|$', 'Interpreter', 'latex', 'FontSize', 11);
            end

            % 输出统计信息
            fprintf('=== Acceleration Analysis ===\n');
            fprintf('Mean acceleration: %.6f\n', mean(accel_norms));
            fprintf('Max acceleration: %.6f\n', max(accel_norms));
            fprintf('Acceleration std: %.6f\n', std(accel_norms));
            if n > 2
                fprintf('Mean acceleration jerk: %.6f\n', mean(abs(accel_smoothness)));
            end
        end

        function plot_error_smoothness_comparison(vs)
            % 误差平滑度量化对比
            if isempty(vs.history)
                warning('No history to plot. Run ibvs.run() first.');
                return;
            end

            % 提取误差平滑度数据
            raw_smoothness = [vs.history.error_smoothness];
            nd_smoothness = [vs.history.nd_error_smoothness];
            raw_errors = [vs.history.en];
            nd_errors = [vs.history.nd_en];

            n = length(vs.history);
            t = 1:n;

            h = figure('Name','Error Smoothness Analysis','NumberTitle','off');

            % 子图1：原始误差vs ND误差
            subplot(2,2,1);
            plot(t, raw_errors, 'b-', 'LineWidth', 1.2, 'DisplayName', 'Raw error');
            hold on;
            plot(t, nd_errors, 'r--', 'LineWidth', 1.2, 'DisplayName', 'ND error');
            grid on;
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
            ylabel('$\|e\|$', 'Interpreter', 'latex', 'FontSize', 11);
            legend('show', 'FontSize', 9);

            % 子图2：误差变化率（平滑度）
            subplot(2,2,2);
            plot(t, raw_smoothness, 'b-', 'LineWidth', 1.2, 'DisplayName', 'Raw $\Delta e$');
            hold on;
            valid_nd = ~isnan(nd_smoothness);
            if any(valid_nd)
                plot(t(valid_nd), nd_smoothness(valid_nd), 'r--', 'LineWidth', 1.2, 'DisplayName', 'ND $\Delta e$');
            end
            grid on;
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
            ylabel('Error change rate', 'FontSize', 11);
            legend('show', 'Interpreter', 'latex', 'FontSize', 9);

            % 子图3：平滑度统计对比
            subplot(2,2,3);
            raw_stats = [mean(raw_smoothness), std(raw_smoothness), max(raw_smoothness)];
            if any(valid_nd)
                nd_stats = [mean(nd_smoothness(valid_nd)), std(nd_smoothness(valid_nd)), max(nd_smoothness(valid_nd))];
                bar_data = [raw_stats; nd_stats]';
                bar(bar_data);
                legend('Raw error', 'ND error', 'Location','best', 'FontSize', 9);
            else
                bar(raw_stats);
            end
            set(gca, 'XTickLabel', {'Mean','Std','Max'});
            ylabel('Smoothness metric', 'FontSize', 11);
            grid on;

            % 子图4：平滑度改善率
            subplot(2,2,4);
            if any(valid_nd)
                improvement = (raw_smoothness(valid_nd) - nd_smoothness(valid_nd)) ./ raw_smoothness(valid_nd) * 100;
                plot(t(valid_nd), improvement, 'g-', 'LineWidth', 1.2);
                grid on;
                xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
                ylabel('Improvement (\%)', 'FontSize', 11);
                mean_improvement = mean(improvement(improvement > -100 & improvement < 100));
                text(0.7, 0.9, sprintf('Mean: %.1f%%', mean_improvement), 'Units', 'normalized');
            end

            % 输出统计信息
            fprintf('=== Error Smoothness Analysis ===\n');
            fprintf('Raw error smoothness: mean=%.6f, std=%.6f\n', mean(raw_smoothness), std(raw_smoothness));
            if any(valid_nd)
                fprintf('ND error smoothness: mean=%.6f, std=%.6f\n', mean(nd_smoothness(valid_nd)), std(nd_smoothness(valid_nd)));
                fprintf('Average smoothness improvement: %.2f%%\n', mean_improvement);
            end
        end

        function plot_trajectory_curvature_analysis(vs)
            % 轨迹曲率分析
            if isempty(vs.history)
                warning('No history to plot. Run ibvs.run() first.');
                return;
            end

            curvatures = [vs.history.curvature];
            positions = [vs.history.Tcam];
            pos_xyz = [positions.t]';  % Nx3矩阵
            n = length(vs.history);
            t = 1:n;

            h = figure('Name','Trajectory Curvature Analysis','NumberTitle','off');

            % 子图1：3D轨迹可视化
            subplot(2,2,1);
            plot3(pos_xyz(:,1), pos_xyz(:,2), pos_xyz(:,3), 'b-', 'LineWidth', 2);
            hold on;
            scatter3(pos_xyz(1,1), pos_xyz(1,2), pos_xyz(1,3), 100, 'g', 'filled');
            scatter3(pos_xyz(end,1), pos_xyz(end,2), pos_xyz(end,3), 100, 'r', 'filled');
            grid on;
            xlabel('$x$ (m)', 'Interpreter', 'latex', 'FontSize', 11);
            ylabel('$y$ (m)', 'Interpreter', 'latex', 'FontSize', 11);
            zlabel('$z$ (m)', 'Interpreter', 'latex', 'FontSize', 11);
            legend('Trajectory', 'Start', 'End', 'Location','best', 'FontSize', 9);

            % 子图2：曲率随时间变化
            subplot(2,2,2);
            plot(t, curvatures, 'r-', 'LineWidth', 1.5);
            grid on;
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
            ylabel('Curvature $\kappa$', 'Interpreter', 'latex', 'FontSize', 11);

            % 子图3：曲率统计分布
            subplot(2,2,3);
            histogram(curvatures, 20);
            xlabel('Curvature $\kappa$', 'Interpreter', 'latex', 'FontSize', 11);
            ylabel('Frequency', 'FontSize', 11);
            grid on;

            % 子图4：路径长度和曲率关系
            subplot(2,2,4);
            if n > 1
                path_segments = sqrt(sum(diff(pos_xyz).^2, 2));
                cumulative_distance = [0; cumsum(path_segments)];
                plot(cumulative_distance, curvatures, 'b-', 'LineWidth', 1.2);
                grid on;
                xlabel('Cumulative distance (m)', 'FontSize', 11);
                ylabel('Curvature $\kappa$', 'Interpreter', 'latex', 'FontSize', 11);
            end

            % 计算轨迹质量指标
            total_distance = cumulative_distance(end);
            straight_distance = norm(pos_xyz(end,:) - pos_xyz(1,:));
            path_efficiency = straight_distance / total_distance * 100;

            % 输出统计信息
            fprintf('=== Trajectory Curvature Analysis ===\n');
            fprintf('Mean curvature: %.6f\n', mean(curvatures));
            fprintf('Max curvature: %.6f\n', max(curvatures));
            fprintf('Curvature std: %.6f\n', std(curvatures));
            fprintf('Total path distance: %.3f\n', total_distance);
            fprintf('Straight line distance: %.3f\n', straight_distance);
            fprintf('Path efficiency: %.2f%%\n', path_efficiency);
        end

        % function plot_nd_vs_no_nd_trajectory(vs)
        %     % Compare actual (ND-run) trajectory vs theoretical no-ND trajectory
        %     if isempty(vs.history)
        %         warning('No history to plot. Run ibvs.run() first.');
        %         return;
        %     end

        %     n = numel(vs.history);
        %     % actual positions
        %     pos_actual = zeros(n,3);
        %     for k=1:n
        %         pos_actual(k,:) = vs.history(k).Tcam.t';
        %     end

        %     % reconstruct hypothetical no-ND trajectory (start from first pose)
        %     T_no_nd = SE3(vs.history(1).Tcam);
        %     pos_no_nd = zeros(n,3);
        %     pos_no_nd(1,:) = T_no_nd.t';

        %     % replay using dt = 1/fps (robust to random Task_process_clcle during run)
        %     dt = 1 / max(vs.fps, 1);

        %     for k = 1:n-1
        %         % raw image features saved in history
        %         uv = reshape(vs.history(k).f, 2, []);
        %         e_raw = uv(:) - vs.uv_star(:);

        %         % attempt Jacobian and velocity command
        %         try
        %             if ~isempty(vs.depth)
        %                 J = vs.camera.visjac_p(uv, vs.depth);
        %             else
        %                 J = vs.camera.visjac_p(uv, 1);
        %             end
        %             v_cmd = -vs.lambda * pinv(J) * e_raw;
        %         catch
        %             v_cmd = zeros(6,1);
        %         end

        %         % integrate same way as controller
        %         v_delta = v_cmd * dt;
        %         Td = SE3(trnorm(delta2tr(v_delta)));
        %         T_no_nd = T_no_nd * Td;
        %         pos_no_nd(k+1,:) = T_no_nd.t';
        %     end

        %     % plot
        %     h = figure('Name','ND vs No-ND Camera Trajectory','NumberTitle','off'); clf;
        %     plot3(pos_actual(:,1), pos_actual(:,2), pos_actual(:,3), 'b-', 'LineWidth', 2); hold on;
        %     plot3(pos_no_nd(:,1), pos_no_nd(:,2), pos_no_nd(:,3), 'r--', 'LineWidth', 1.5);
        %     scatter3(pos_actual(1,1), pos_actual(1,2), pos_actual(1,3), 80, 'go', 'filled');
        %     scatter3(pos_actual(end,1), pos_actual(end,2), pos_actual(end,3), 80, 'mo', 'filled');
        %     legend('Actual (ND run)', 'Predicted no-ND', 'Start', 'End', 'Location', 'best');
        %     xlabel('X'); ylabel('Y'); zlabel('Z'); grid on; axis equal; title('ND vs predicted no-ND trajectories');
        %     view(3);

        %     % simple numeric comparison
        %     dists = sqrt(sum((pos_actual - pos_no_nd).^2, 2));
        %     fprintf('No-ND vs ND: mean separation=%.6f, max=%.6f, final separation=%.6f\n', ...
        %         mean(dists), max(dists), dists(end));
        % end

        function plot_p_nd_comparison(vs)
            % Compare feature trajectories: actual (ND-run) vs theoretical no-ND
            if isempty(vs.history)
                warning('No history to plot. Run ibvs.run() first.');
                return;
            end
            if strcmp(vs.type, 'point') ~= 1
                warning('Feature trajectory comparison only for point-based IBVS');
                return;
            end

            n = numel(vs.history);
            npts = size(vs.uv_star, 2);  % number of feature points

            % extract actual feature trajectory (with ND if enabled)
            uv_actual = [vs.history.f]';  % Nx(2*npts)

            % reconstruct hypothetical no-ND feature trajectory
            uv_no_nd = zeros(n, 2*npts);
            T_no_nd = SE3(vs.history(1).Tcam);
            dt = 1 / max(vs.fps, 1);

            for k = 1:n
                % project world points using hypothetical camera pose
                try
                    uv_k = vs.camera.project(vs.P, 'Tcam', T_no_nd);
                    uv_no_nd(k,:) = uv_k(:)';
                catch
                    uv_no_nd(k,:) = nan;
                end

                % compute next pose (skip last step)
                if k < n
                    uv_2col = reshape(vs.history(k).f, 2, []);
                    e_raw = uv_2col(:) - vs.uv_star(:);

                    try
                        if ~isempty(vs.depth)
                            J = vs.camera.visjac_p(uv_2col, vs.depth);
                        else
                            J = vs.camera.visjac_p(uv_2col, 1);
                        end
                        v_cmd = -vs.lambda * pinv(J) * e_raw;
                    catch
                        v_cmd = zeros(6,1);
                    end

                    v_delta = v_cmd * dt;
                    Td = SE3(trnorm(delta2tr(v_delta)));
                    T_no_nd = T_no_nd * Td;
                end
            end

            % plot with HIGH CONTRAST colors
            h = figure('Name','Feature Trajectory: ND vs No-ND','NumberTitle','off'); clf;
            hold on;

            % 高对比度颜色：ND用深色（蓝、红、绿、品红），no-ND用亮色（青、黄、橙、粉）
            colors_nd = [0 0 0.8; 0.8 0 0; 0 0.6 0; 0.7 0 0.7];      % 深蓝、深红、深绿、深紫
            colors_no_nd = [0 0.9 0.9; 1 0.8 0; 1 0.5 0; 1 0.4 0.7]; % 青、黄、橙、粉

            for i = 1:npts
                % 实际轨迹（启用ND后）- 深色粗实线
                u_act = uv_actual(:,2*i-1); 
                v_act = uv_actual(:,2*i);
                plot(u_act, v_act, '-', 'Color', colors_nd(i,:), 'LineWidth', 3, ...
                     'DisplayName', sprintf('Pt%d (实际-ND启用)', i));

                % 预测轨迹（理论不用ND）- 亮色虚线+方形标记
                u_no = uv_no_nd(:,2*i-1); 
                v_no = uv_no_nd(:,2*i);
                plot(u_no, v_no, '--', 'Color', colors_no_nd(i,:), 'LineWidth', 2.5, ...
                     'DisplayName', sprintf('Pt%d (预测-无ND)', i));

                % 每隔n/10点加方形标记
                idxm = 1:max(1,round(n/10)):n;
                plot(u_no(idxm), v_no(idxm), 's', 'MarkerEdgeColor', colors_no_nd(i,:), ...
                     'MarkerFaceColor', colors_no_nd(i,:), 'MarkerSize', 8);
            end

            % mark initial positions
            plot_poly(reshape(uv_actual(1,:), 2, []), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');

            % mark desired positions
            plot_poly(vs.uv_star, 'r^', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

            % mark final positions
            plot_poly(reshape(uv_actual(end,:), 2, []), 'bs', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
            plot_poly(reshape(uv_no_nd(end,:), 2, []), 'ms', 'MarkerSize', 8);

            axis([0 vs.camera.npix(1) 0 vs.camera.npix(2)]);
            daspect([1 1 1]);
            set(gca, 'Ydir', 'reverse');
            grid on;
            xlabel('u (pixels)');
            ylabel('v (pixels)');
            title('Feature Trajectories: ND-enabled vs No-ND prediction');
            legend('Location', 'bestoutside');
            hold off;

            % compute trajectory differences
            diff_final = sqrt(sum((uv_actual(end,:) - uv_no_nd(end,:)).^2));
            diff_mean = mean(sqrt(sum((uv_actual - uv_no_nd).^2, 2)));

            fprintf('\n=== 特征轨迹对比（实际ND启用 vs 理论无ND）===\n');
            fprintf('【验证】实际轨迹来自：vs.history.f（当前运行记录，包含ND效应）\n');
            fprintf('【验证】预测轨迹来自：用原始误差e_raw重新计算的控制律（模拟无ND情况）\n');
            fprintf('平均轨迹偏差: %.3f 像素\n', diff_mean);
            fprintf('最终位置偏差: %.3f 像素\n', diff_final);

            % per-point statistics
            fprintf('\n各特征点详细偏差统计:\n');
            for i = 1:npts
                u_diff = uv_actual(:, 2*i-1) - uv_no_nd(:, 2*i-1);
                v_diff = uv_actual(:, 2*i) - uv_no_nd(:, 2*i);
                pt_diff = sqrt(u_diff.^2 + v_diff.^2);
                fprintf('  特征点%d: 平均=%.3f, 最大=%.3f 像素\n', i, mean(pt_diff), max(pt_diff));
            end
            fprintf('\n【解读】若偏差很小(<1像素)说明ND效果不明显或两条轨迹重合；\n');
            fprintf('        若偏差较大(>5像素)说明ND显著改变了轨迹形状。\n');
        end

              function plot_nd_velocity_comparison(vs)
            % 绘制线速度和角速度的对比（加入ND前后）
            % 展示ND对抗采样抖动(Jitter)的效果：使用等效标准速度进行对比
            if isempty(vs.history)
                warning('No history to plot. Run ibvs.run() first.');
                return;
            end

            n = length(vs.history);
            t = 1:n;

            % 计算标准参考时间步长 (Nominal Time Step)
            dt_nominal = 1 / max(vs.fps, 1);

            raw_vel_eq = zeros(6, n);
            nd_vel_eq = zeros(6, n);

            for k=1:n
                % 获取实际的时间步长 (包含随机抖动)
                if isfield(vs.history(k), 'dt') && ~isempty(vs.history(k).dt)
                   dt_actual = vs.history(k).dt;
                else
                   dt_actual = dt_nominal; 
                end

                dt_nominal = 1 / max(vs.fps, 1);

                % 1. 原始等效速度 (Raw Effective Velocity / Original End-effector Velocity)
                % 定义：
                % - 这不是视觉伺服直接计算出的速度命令 (v_cmd)。
                % - 这是受变帧率(varying frame rate)影响后的实际末端速度。
                % - 物理含义：如果直接执行 "v_cmd * dt_actual" 的位移，在标准时间轴下体现出的速度冲击。
                if isfield(vs.history(k), 'raw_vel')
                    v_cmd = vs.history(k).raw_vel;
                    raw_vel_eq(:, k) = v_cmd * (dt_actual / dt_nominal);
                else
                    raw_vel_eq(:, k) = zeros(6,1);
                end

                % 2. ND等效速度 (ND Smoothed Equivalent Velocity)
                % - 这是经过ND平滑实际执行位移后的等效速度。
                step_nd = vs.history(k).vel;
                nd_vel_eq(:, k) = step_nd / dt_nominal;
            end

            % --- 1. 线速度对比 (Linear Velocity) ---
            figure('Name', 'Linear Velocity Comparison (ND Effect)', 'NumberTitle', 'off', 'Color', 'w');
            labels = {'$v_x$ (m/s)', '$v_y$ (m/s)', '$v_z$ (m/s)'};
            for i=1:3
                subplot(3,1,i);
                plot(t, raw_vel_eq(i,:), 'b-', 'LineWidth', 0.8, 'DisplayName', 'Raw equiv. velocity');
                hold on;
                plot(t, nd_vel_eq(i,:), 'r-', 'LineWidth', 1.2, 'DisplayName', 'ND equiv. velocity');

                ylabel(labels{i}, 'Interpreter', 'latex', 'FontSize', 11); grid on;
                set(gca, 'FontSize', 10);
                if i==1
                    legend('show', 'Location', 'northeast', 'FontSize', 9);
                end
            end
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);

            % --- 2. 角速度对比 (Angular Velocity) ---
            figure('Name', 'Angular Velocity Comparison (ND Effect)', 'NumberTitle', 'off', 'Color', 'w');
            labels = {'$\omega_x$ (rad/s)', '$\omega_y$ (rad/s)', '$\omega_z$ (rad/s)'};
            for i=1:3
                idx = i + 3; % 4, 5, 6
                subplot(3,1,i);
                plot(t, raw_vel_eq(idx,:), 'b-', 'LineWidth', 0.8, 'DisplayName', 'Raw equiv. velocity');
                hold on;
                plot(t, nd_vel_eq(idx,:), 'r-', 'LineWidth', 1.2, 'DisplayName', 'ND equiv. velocity');

                ylabel(labels{i}, 'Interpreter', 'latex', 'FontSize', 11); grid on;
                set(gca, 'FontSize', 10);
                if i==1
                    legend('show', 'Location', 'northeast', 'FontSize', 9);
                end
            end
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
        end

        function plot_nd_acceleration_comparison(vs)
            % 绘制线加速度和角加速度的对比（加入ND前后）
            % 展示抖动环境下的等效加速度对比
            if isempty(vs.history)
                warning('No history to plot. Run ibvs.run() first.');
                return;
            end

            n = length(vs.history);
            t = 1:n;

            % 计算标准参考时间步长 (Nominal Time Step)
            dt_nominal = 1 / max(vs.fps, 1);

            nd_accel_eq = zeros(6, n);
            raw_accel_eq = zeros(6, n);

            % 重新计算等效速度历史 (Recompute Effective Velocities)
            v_raw_eq_hist = zeros(6, n);
            v_nd_eq_hist = zeros(6, n);

             for k=1:n
                if isfield(vs.history(k), 'dt') && ~isempty(vs.history(k).dt)
                   dt_act = vs.history(k).dt; 
                else
                   dt_act = dt_nominal; 
                end

                % 1. 原始等效速度 (Original Effective Velocity)
                % 定义: 相机末端速度，受变帧率影响 (v_cmd * dt_act / dt_nominal)
                if isfield(vs.history(k), 'raw_vel')
                    v_raw_eq_hist(:, k) = vs.history(k).raw_vel * (dt_act / dt_nominal);
                end

                % 2. ND等效速度 (ND Effective Velocity)
                v_nd_eq_hist(:, k) = vs.history(k).vel / dt_nominal;
            end

            % 差分计算等效加速度 (Equivalent Acceleration)
            % 与等效速度保持一致，分母使用 dt_nominal（均匀时间网格）
            for k=2:n
                raw_accel_eq(:, k) = (v_raw_eq_hist(:, k) - v_raw_eq_hist(:, k-1)) / dt_nominal;
                nd_accel_eq(:, k) = (v_nd_eq_hist(:, k) - v_nd_eq_hist(:, k-1)) / dt_nominal;
            end

            % --- 1. 线加速度对比 (Linear Acceleration) ---
            figure('Name', 'Linear Acceleration Comparison (ND Effect)', 'NumberTitle', 'off', 'Color', 'w');
            labels = {'$a_x$ (m/s$^2$)', '$a_y$ (m/s$^2$)', '$a_z$ (m/s$^2$)'};
            for i=1:3
                subplot(3,1,i);
                plot(t, raw_accel_eq(i,:), 'b-', 'LineWidth', 0.8, 'DisplayName', 'Raw equiv. accel.');
                hold on;
                plot(t, nd_accel_eq(i,:), 'r-', 'LineWidth', 1.2, 'DisplayName', 'ND equiv. accel.');

                ylabel(labels{i}, 'Interpreter', 'latex', 'FontSize', 11); grid on;
                set(gca, 'FontSize', 10);
                if i==1
                    legend('show', 'Location', 'northeast', 'FontSize', 9);
                end
            end
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);

            % --- 2. 角加速度对比 (Angular Acceleration) ---
            figure('Name', 'Angular Acceleration Comparison (ND Effect)', 'NumberTitle', 'off', 'Color', 'w');
            labels = {'$\dot{\omega}_x$ (rad/s$^2$)', '$\dot{\omega}_y$ (rad/s$^2$)', '$\dot{\omega}_z$ (rad/s$^2$)'};
            for i=1:3
                idx = i + 3; % 4, 5, 6
                subplot(3,1,i);
                plot(t, raw_accel_eq(idx,:), 'b-', 'LineWidth', 0.8, 'DisplayName', 'Raw equiv. accel.');
                hold on;
                plot(t, nd_accel_eq(idx,:), 'r-', 'LineWidth', 1.2, 'DisplayName', 'ND equiv. accel.');

                ylabel(labels{i}, 'Interpreter', 'latex', 'FontSize', 11); grid on;
                set(gca, 'FontSize', 10);
                if i==1
                    legend('show', 'Location', 'northeast', 'FontSize', 9);
                end
            end
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
        end


        function plot_true_velocity(vs)
            % 绘制控制律输出的真实速度 v_cmd（非等效速度）
            % 验证 v_cmd 本身是否平滑
            if isempty(vs.history)
                warning('No history to plot. Run ibvs.run() first.');
                return;
            end

            n = length(vs.history);
            t = 1:n;
            dt_nominal = 1 / max(vs.fps, 1);

            v_true = zeros(6, n);     % 真实速度 v_cmd
            v_equiv = zeros(6, n);    % 等效速度 v_cmd * (dt_k / dt_nom)

            for k=1:n
                if isfield(vs.history(k), 'raw_vel') && ~isempty(vs.history(k).raw_vel)
                    v_true(:, k) = vs.history(k).raw_vel;
                end
                if isfield(vs.history(k), 'dt') && ~isempty(vs.history(k).dt)
                    dt_actual = vs.history(k).dt;
                else
                    dt_actual = dt_nominal;
                end
                v_equiv(:, k) = v_true(:, k) * (dt_actual / dt_nominal);
            end

            % --- 线速度 ---
            figure('Name', 'True v_cmd vs Equivalent Velocity', 'NumberTitle', 'off', 'Color', 'w');
            labels = {'$v_x$ (m/s)', '$v_y$ (m/s)', '$v_z$ (m/s)'};
            for i=1:3
                subplot(3,1,i);
                plot(t, v_true(i,:), 'k-', 'LineWidth', 1.5, 'DisplayName', 'True $\mathbf{v}_{\mathrm{cmd}}$');
                hold on;
                plot(t, v_equiv(i,:), 'b-', 'LineWidth', 0.8, 'DisplayName', 'Equiv. $\mathbf{v}_{\mathrm{cmd}} \cdot \Delta t_k / \Delta t_{\mathrm{nom}}$');
                ylabel(labels{i}, 'Interpreter', 'latex', 'FontSize', 11); grid on;
                set(gca, 'FontSize', 10);
                if i==1
                    legend('show', 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 9);
                end
            end
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);

            % --- 角速度 ---
            figure('Name', 'True \omega_{cmd} vs Equivalent Angular Velocity', 'NumberTitle', 'off', 'Color', 'w');
            labels = {'$\omega_x$ (rad/s)', '$\omega_y$ (rad/s)', '$\omega_z$ (rad/s)'};
            for i=1:3
                idx = i + 3;
                subplot(3,1,i);
                plot(t, v_true(idx,:), 'k-', 'LineWidth', 1.5, 'DisplayName', 'True $\boldsymbol{\omega}_{\mathrm{cmd}}$');
                hold on;
                plot(t, v_equiv(idx,:), 'b-', 'LineWidth', 0.8, 'DisplayName', 'Equiv. angular vel.');
                ylabel(labels{i}, 'Interpreter', 'latex', 'FontSize', 11); grid on;
                set(gca, 'FontSize', 10);
                if i==1
                    legend('show', 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 9);
                end
            end
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
        end


        function plot_nd_displacement_comparison(vs)
            % 绘制位移增量对比（加入ND前后）
            % 展示ND对原始位移的平滑效果：直接对比每步位移增量
            % Raw displacement: v_cmd * dt_k (未经ND的原始位移)
            % ND displacement:  x^{k+1} (经ND平滑后的实际执行位移)
            if isempty(vs.history)
                warning('No history to plot. Run ibvs.run() first.');
                return;
            end

            n = length(vs.history);
            t = 1:n;

            raw_disp = zeros(6, n);
            nd_disp  = zeros(6, n);

            for k=1:n
                % 获取实际时间步长
                if isfield(vs.history(k), 'dt') && ~isempty(vs.history(k).dt)
                    dt_actual = vs.history(k).dt;
                else
                    dt_actual = 1 / max(vs.fps, 1);
                end

                % 原始位移增量: v_cmd * dt_k
                if isfield(vs.history(k), 'raw_vel')
                    raw_disp(:, k) = vs.history(k).raw_vel * dt_actual;
                end

                % ND平滑后的位移增量: x^{k+1} (hist.vel 即为实际执行位移)
                nd_disp(:, k) = vs.history(k).vel;
            end

            % --- 1. 线位移增量对比 (Linear Displacement Increment) ---
            figure('Name', 'Linear Displacement Comparison (ND Effect)', 'NumberTitle', 'off', 'Color', 'w');
            labels = {'$\Delta p_x$ (m)', '$\Delta p_y$ (m)', '$\Delta p_z$ (m)'};
            for i=1:3
                subplot(3,1,i);
                plot(t, raw_disp(i,:), 'b-', 'LineWidth', 0.8, 'DisplayName', 'Raw displacement');
                hold on;
                plot(t, nd_disp(i,:), 'r-', 'LineWidth', 1.2, 'DisplayName', 'ND displacement');

                ylabel(labels{i}, 'Interpreter', 'latex', 'FontSize', 11); grid on;
                set(gca, 'FontSize', 10);
                if i==1
                    legend('show', 'Location', 'northeast', 'FontSize', 9);
                end
            end
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);

            % --- 2. 角位移增量对比 (Angular Displacement Increment) ---
            figure('Name', 'Angular Displacement Comparison (ND Effect)', 'NumberTitle', 'off', 'Color', 'w');
            labels = {'$\Delta \theta_x$ (rad)', '$\Delta \theta_y$ (rad)', '$\Delta \theta_z$ (rad)'};
            for i=1:3
                idx = i + 3;
                subplot(3,1,i);
                plot(t, raw_disp(idx,:), 'b-', 'LineWidth', 0.8, 'DisplayName', 'Raw displacement');
                hold on;
                plot(t, nd_disp(idx,:), 'r-', 'LineWidth', 1.2, 'DisplayName', 'ND displacement');

                ylabel(labels{i}, 'Interpreter', 'latex', 'FontSize', 11); grid on;
                set(gca, 'FontSize', 10);
                if i==1
                    legend('show', 'Location', 'northeast', 'FontSize', 9);
                end
            end
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
        end


        function [history_raw, history_nd] = run_dual_experiment(vs, seed)
            % 双实验对比：相同随机采样序列下，分别运行 无ND / 有ND
            % 用于客观对比 ND 对速度、位移、加速度的真实影响
            %
            % 用法: [h_raw, h_nd] = vs.run_dual_experiment(42);
            %       vs.plot_dual_velocity(h_raw, h_nd);
            %       vs.plot_dual_displacement(h_raw, h_nd);
            %       vs.plot_dual_acceleration(h_raw, h_nd);

            if nargin < 2, seed = 42; end

            % 获取最大迭代次数
            if ~isempty(vs.niter)
                max_iter = vs.niter;
            else
                max_iter = 500;
            end

            % 预生成随机采样序列（两组实验使用完全相同的 dt 序列）
            rng(seed);
            dt_base_ms = round(1/vs.fps * 1000);
            dt_min = vs.Task_process_cycle_min;
            dt_max = vs.Task_process_cycle_max;
            dt_seq = zeros(max_iter, 1);
            for k = 1:max_iter
                range = (dt_max - dt_min) * rand(1,1) + dt_min;
                dt_seq(k) = round(dt_base_ms + range);
            end

            orig_nd = vs.nd_enabled;

            % ========== 实验 A: 无 ND（基线） ==========
            fprintf('=== Experiment A: Without ND (Baseline) ===\n');
            vs.nd_enabled = false;
            vs.nd_pos_step_state = [];
            vs.pf_s_ref = [];
            vs.init();

            for k = 1:max_iter
                vs.Task_process_clcle = dt_seq(k);
                status = vs.step();
                drawnow
                if status ~= 0, break; end
            end
            history_raw = vs.history;
            fprintf('  Completed: %d steps\n', length(history_raw));

            % ========== 实验 B: 有 ND ==========
            fprintf('=== Experiment B: With ND ===\n');
            vs.nd_enabled = true;
            vs.nd_pos_step_state = [];
            vs.pf_s_ref = [];
            vs.init();

            for k = 1:max_iter
                vs.Task_process_clcle = dt_seq(k);
                status = vs.step();
                drawnow
                if status ~= 0, break; end
            end
            history_nd = vs.history;
            fprintf('  Completed: %d steps\n', length(history_nd));

            % 恢复原始设置
            vs.nd_enabled = orig_nd;

            % 保存到对象，供后续单独调用绘图函数
            vs.dual_history_raw = history_raw;
            vs.dual_history_nd  = history_nd;

            % Auto-plot
            vs.plot_dual_velocity(history_raw, history_nd);
            vs.plot_dual_displacement(history_raw, history_nd);
            vs.plot_dual_acceleration(history_raw, history_nd);
            vs.plot_dual_velocity_analysis(history_raw, history_nd);
        end


        function plot_dual_velocity(vs, history_raw, history_nd)
            % 双实验速度对比：真实 v_cmd（控制律输出）
            % 用法: vs.plot_dual_velocity()           — 使用已保存的双实验结果
            %       vs.plot_dual_velocity(h_raw, h_nd) — 手动传入
            if nargin < 3
                history_raw = vs.dual_history_raw;
                history_nd  = vs.dual_history_nd;
            end
            if isempty(history_raw) || isempty(history_nd)
                error('No dual experiment data. Run vs.run_dual_experiment() first.');
            end
            n = min(length(history_raw), length(history_nd));
            t = 1:n;

            v_raw = zeros(6, n);
            v_nd  = zeros(6, n);
            for k = 1:n
                v_raw(:, k) = history_raw(k).raw_vel;
                v_nd(:, k)  = history_nd(k).raw_vel;
            end

            % --- 线速度 ---
            figure('Name', 'Dual Exp: Linear Velocity (v_{cmd})', 'NumberTitle', 'off', 'Color', 'w');
            labels = {'$v_x$ (m/s)', '$v_y$ (m/s)', '$v_z$ (m/s)'};
            for i = 1:3
                subplot(3,1,i);
                plot(t, v_raw(i,:), 'b-', 'LineWidth', 0.8, 'DisplayName', 'Baseline (w/o $\mathcal{G}_t$)');
                hold on;
                plot(t, v_nd(i,:), 'r-', 'LineWidth', 1.2, 'DisplayName', 'PF-NDRG (w/ $\mathcal{G}_t$)');
                ylabel(labels{i}, 'Interpreter', 'latex', 'FontSize', 11);
                grid on; set(gca, 'FontSize', 10);
                if i == 1
                    legend('show', 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 9);
                end
            end
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);

            % --- 角速度 ---
            figure('Name', 'Dual Exp: Angular Velocity (\omega_{cmd})', 'NumberTitle', 'off', 'Color', 'w');
            labels = {'$\omega_x$ (rad/s)', '$\omega_y$ (rad/s)', '$\omega_z$ (rad/s)'};
            for i = 1:3
                idx = i + 3;
                subplot(3,1,i);
                plot(t, v_raw(idx,:), 'b-', 'LineWidth', 0.8, 'DisplayName', 'Baseline (w/o $\mathcal{G}_t$)');
                hold on;
                plot(t, v_nd(idx,:), 'r-', 'LineWidth', 1.2, 'DisplayName', 'PF-NDRG (w/ $\mathcal{G}_t$)');
                ylabel(labels{i}, 'Interpreter', 'latex', 'FontSize', 11);
                grid on; set(gca, 'FontSize', 10);
                if i == 1
                    legend('show', 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 9);
                end
            end
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
        end


        function plot_dual_displacement(vs, history_raw, history_nd)
            % 双实验位移对比：真实执行的位移增量
            % 用法: vs.plot_dual_displacement()           — 使用已保存的双实验结果
            %       vs.plot_dual_displacement(h_raw, h_nd) — 手动传入
            if nargin < 3
                history_raw = vs.dual_history_raw;
                history_nd  = vs.dual_history_nd;
            end
            if isempty(history_raw) || isempty(history_nd)
                error('No dual experiment data. Run vs.run_dual_experiment() first.');
            end
            n = min(length(history_raw), length(history_nd));
            t = 1:n;

            d_raw = zeros(6, n);
            d_nd  = zeros(6, n);
            for k = 1:n
                d_raw(:, k) = history_raw(k).vel;
                d_nd(:, k)  = history_nd(k).vel;
            end

            % --- 线位移 ---
            figure('Name', 'Dual Exp: Linear Displacement', 'NumberTitle', 'off', 'Color', 'w');
            labels = {'$\Delta p_x$ (m)', '$\Delta p_y$ (m)', '$\Delta p_z$ (m)'};
            for i = 1:3
                subplot(3,1,i);
                plot(t, d_raw(i,:), 'b-', 'LineWidth', 0.8, 'DisplayName', 'Baseline (w/o $\mathcal{G}_t$)');
                hold on;
                plot(t, d_nd(i,:), 'r-', 'LineWidth', 1.2, 'DisplayName', 'PF-NDRG (w/ $\mathcal{G}_t$)');
                ylabel(labels{i}, 'Interpreter', 'latex', 'FontSize', 11);
                grid on; set(gca, 'FontSize', 10);
                if i == 1
                    legend('show', 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 9);
                end
            end
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);

            % --- 角位移 ---
            figure('Name', 'Dual Exp: Angular Displacement', 'NumberTitle', 'off', 'Color', 'w');
            labels = {'$\Delta \theta_x$ (rad)', '$\Delta \theta_y$ (rad)', '$\Delta \theta_z$ (rad)'};
            for i = 1:3
                idx = i + 3;
                subplot(3,1,i);
                plot(t, d_raw(idx,:), 'b-', 'LineWidth', 0.8, 'DisplayName', 'Baseline (w/o $\mathcal{G}_t$)');
                hold on;
                plot(t, d_nd(idx,:), 'r-', 'LineWidth', 1.2, 'DisplayName', 'PF-NDRG (w/ $\mathcal{G}_t$)');
                ylabel(labels{i}, 'Interpreter', 'latex', 'FontSize', 11);
                grid on; set(gca, 'FontSize', 10);
                if i == 1
                    legend('show', 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 9);
                end
            end
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
        end


        function plot_dual_acceleration(vs, history_raw, history_nd)
            % 双实验加速度对比：物理加速度 = Δv_cmd / dt_k
            % 用法: vs.plot_dual_acceleration()           — 使用已保存的双实验结果
            %       vs.plot_dual_acceleration(h_raw, h_nd) — 手动传入
            if nargin < 3
                history_raw = vs.dual_history_raw;
                history_nd  = vs.dual_history_nd;
            end
            if isempty(history_raw) || isempty(history_nd)
                error('No dual experiment data. Run vs.run_dual_experiment() first.');
            end
            n = min(length(history_raw), length(history_nd));
            t = 1:n;

            a_raw = zeros(6, n);
            a_nd  = zeros(6, n);

            for k = 2:n
                dt_k = history_raw(k).dt;
                a_raw(:, k) = (history_raw(k).raw_vel - history_raw(k-1).raw_vel) / dt_k;
                dt_k_nd = history_nd(k).dt;
                a_nd(:, k)  = (history_nd(k).raw_vel  - history_nd(k-1).raw_vel)  / dt_k_nd;
            end

            % --- 线加速度 ---
            figure('Name', 'Dual Exp: Linear Acceleration', 'NumberTitle', 'off', 'Color', 'w');
            labels = {'$a_x$ (m/s$^2$)', '$a_y$ (m/s$^2$)', '$a_z$ (m/s$^2$)'};
            for i = 1:3
                subplot(3,1,i);
                plot(t, a_raw(i,:), 'b-', 'LineWidth', 0.8, 'DisplayName', 'Baseline (w/o $\mathcal{G}_t$)');
                hold on;
                plot(t, a_nd(i,:), 'r-', 'LineWidth', 1.2, 'DisplayName', 'PF-NDRG (w/ $\mathcal{G}_t$)');
                ylabel(labels{i}, 'Interpreter', 'latex', 'FontSize', 11);
                grid on; set(gca, 'FontSize', 10);
                if i == 1
                    legend('show', 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 9);
                end
            end
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);

            % --- 角加速度 ---
            figure('Name', 'Dual Exp: Angular Acceleration', 'NumberTitle', 'off', 'Color', 'w');
            labels = {'$\dot{\omega}_x$ (rad/s$^2$)', '$\dot{\omega}_y$ (rad/s$^2$)', '$\dot{\omega}_z$ (rad/s$^2$)'};
            for i = 1:3
                idx = i + 3;
                subplot(3,1,i);
                plot(t, a_raw(idx,:), 'b-', 'LineWidth', 0.8, 'DisplayName', 'Baseline (w/o $\mathcal{G}_t$)');
                hold on;
                plot(t, a_nd(idx,:), 'r-', 'LineWidth', 1.2, 'DisplayName', 'PF-NDRG (w/ $\mathcal{G}_t$)');
                ylabel(labels{i}, 'Interpreter', 'latex', 'FontSize', 11);
                grid on; set(gca, 'FontSize', 10);
                if i == 1
                    legend('show', 'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 9);
                end
            end
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
        end


        function plot_dual_velocity_analysis(vs, history_raw, history_nd)
            % Comprehensive velocity comparison analysis for Experiment 2.
            % Demonstrates that:
            %   (1) Both velocity profiles are smooth (velocity is NOT the jitter source)
            %   (2) PF-NDRG velocity converges slightly slower due to indirect feedback
            %   (3) The indirect feedback chain: bounded Dp -> perturbed s -> perturbed e -> perturbed v
            %
            % Generates:
            %   Figure 1: Velocity comparison (3x2 subplots: v_x..v_z, w_x..w_z)
            %   Figure 2: Velocity vs displacement jitter contrast
            %   Figure 3: Indirect feedback chain visualization
            %   Console:  Quantitative metrics table
            %
            % Usage: vs.plot_dual_velocity_analysis()
            %        vs.plot_dual_velocity_analysis(h_raw, h_nd)

            if nargin < 3
                history_raw = vs.dual_history_raw;
                history_nd  = vs.dual_history_nd;
            end
            if isempty(history_raw) || isempty(history_nd)
                error('No dual experiment data. Run vs.run_dual_experiment() first.');
            end

            n = min(length(history_raw), length(history_nd));
            t = 1:n;

            % --- Extract data ---
            v_raw = zeros(6, n);  v_nd  = zeros(6, n);
            d_raw = zeros(6, n);  d_nd  = zeros(6, n);
            e_raw = zeros(8, n);  e_nd  = zeros(8, n);
            f_raw = zeros(8, n);  f_nd  = zeros(8, n);
            dt_seq = zeros(1, n);
            for k = 1:n
                v_raw(:, k) = history_raw(k).raw_vel;
                v_nd(:, k)  = history_nd(k).raw_vel;
                d_raw(:, k) = history_raw(k).vel;
                d_nd(:, k)  = history_nd(k).vel;
                e_raw(:, k) = history_raw(k).e;
                e_nd(:, k)  = history_nd(k).e;
                f_raw(:, k) = history_raw(k).f;
                f_nd(:, k)  = history_nd(k).f;
                dt_seq(k)   = history_raw(k).dt;
            end

            % ====================================================================
            %  Figure 1: Velocity comparison (paper figure: fig:velocity_comparison)
            %  Layout: 3x2, left = linear v, right = angular w
            % ====================================================================
            figure('Name', 'Velocity Command Comparison', 'NumberTitle', 'off', ...
                   'Color', 'w', 'Position', [100 100 900 600]);

            lin_labels = {'$v_x$ (m/s)', '$v_y$ (m/s)', '$v_z$ (m/s)'};
            ang_labels = {'$\omega_x$ (rad/s)', '$\omega_y$ (rad/s)', '$\omega_z$ (rad/s)'};

            for i = 1:3
                % Linear velocity (left column)
                subplot(3, 2, 2*i-1);
                plot(t, v_raw(i,:), 'b-', 'LineWidth', 0.8);
                hold on;
                plot(t, v_nd(i,:), 'r-', 'LineWidth', 1.2);
                ylabel(lin_labels{i}, 'Interpreter', 'latex', 'FontSize', 11);
                grid on; set(gca, 'FontSize', 10);
                if i == 1
                    legend('Baseline (w/o $\mathcal{G}_t$)', 'PF-NDRG (w/ $\mathcal{G}_t$)', ...
                           'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 9);
                end
                if i == 3
                    xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
                end

                % Angular velocity (right column)
                subplot(3, 2, 2*i);
                plot(t, v_raw(i+3,:), 'b-', 'LineWidth', 0.8);
                hold on;
                plot(t, v_nd(i+3,:), 'r-', 'LineWidth', 1.2);
                ylabel(ang_labels{i}, 'Interpreter', 'latex', 'FontSize', 11);
                grid on; set(gca, 'FontSize', 10);
                if i == 1
                    legend('Baseline (w/o $\mathcal{G}_t$)', 'PF-NDRG (w/ $\mathcal{G}_t$)', ...
                           'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 9);
                end
                if i == 3
                    xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
                end
            end

            % ====================================================================
            %  Figure 2: Velocity jitter vs Displacement jitter (contrast)
            %  Key claim: velocity is smooth, displacement is jittery
            % ====================================================================
            figure('Name', 'Velocity vs Displacement Jitter Contrast', ...
                   'NumberTitle', 'off', 'Color', 'w', 'Position', [100 100 900 700]);

            % Compute step-to-step changes (jitter measure)
            dv_raw = diff(v_raw, 1, 2);  % velocity change per step
            dd_raw = diff(d_raw, 1, 2);  % displacement change per step

            % Top row: velocity step-to-step change (should be smooth)
            for i = 1:3
                subplot(4, 2, 2*i-1);
                plot(2:n, dv_raw(i,:), 'b-', 'LineWidth', 0.6);
                ylabel(['$\Delta v_', num2str(i), '$ (m/s)'], 'Interpreter', 'latex', 'FontSize', 10);
                grid on; set(gca, 'FontSize', 9);
                if i == 1
                    text(0.5, 1.15, '\textbf{Velocity step-to-step change} (smooth)', ...
                         'Units', 'normalized', 'HorizontalAlignment', 'center', ...
                         'Interpreter', 'latex', 'FontSize', 11);
                end
            end

            % Right column: displacement step-to-step change (should be jittery)
            for i = 1:3
                subplot(4, 2, 2*i);
                plot(2:n, dd_raw(i,:), 'b-', 'LineWidth', 0.6);
                ylabel(['$\Delta(\Delta p_', num2str(i), ')$ (m)'], 'Interpreter', 'latex', 'FontSize', 10);
                grid on; set(gca, 'FontSize', 9);
                if i == 1
                    text(0.5, 1.15, '\textbf{Displacement step-to-step change} (jittery)', ...
                         'Units', 'normalized', 'HorizontalAlignment', 'center', ...
                         'Interpreter', 'latex', 'FontSize', 11);
                end
            end

            % Bottom: sampling period overlay
            subplot(4, 2, [7 8]);
            plot(t, dt_seq * 1000, 'k-', 'LineWidth', 0.6);
            ylabel('$\Delta t_k$ (ms)', 'Interpreter', 'latex', 'FontSize', 11);
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
            grid on; set(gca, 'FontSize', 10);
            yline(mean(dt_seq)*1000, 'r--', 'LineWidth', 1.0);
            legend('$\Delta t_k$', '$\overline{\Delta t}$', ...
                   'Interpreter', 'latex', 'FontSize', 9, 'Location', 'northeast');

            % ====================================================================
            %  Figure 3: Indirect feedback chain visualization
            %  Chain: bounded Dp_k -> different T_{c,k+1} -> different s_{k+1}
            %         -> different e_{k+1} -> different v_{c,k+1}
            % ====================================================================
            figure('Name', 'Indirect Feedback Chain Analysis', ...
                   'NumberTitle', 'off', 'Color', 'w', 'Position', [100 100 900 700]);

            % (a) Displacement difference: |Dp_raw - Dp_nd|
            dp_diff = vecnorm(d_raw - d_nd, 2, 1);
            subplot(4, 1, 1);
            plot(t, dp_diff, 'Color', [0.0 0.5 0.0], 'LineWidth', 1.0);
            ylabel('$\|\Delta\mathbf{p}^{\mathrm{raw}} - \Delta\mathbf{p}^{\mathrm{ND}}\|$', ...
                   'Interpreter', 'latex', 'FontSize', 11);
            grid on; set(gca, 'FontSize', 10);
            text(0.02, 0.85, '(a) Displacement difference', 'Units', 'normalized', ...
                 'Interpreter', 'latex', 'FontSize', 10);

            % (b) Feature observation difference: |s_raw - s_nd|
            fs_diff = vecnorm(f_raw - f_nd, 2, 1);
            subplot(4, 1, 2);
            plot(t, fs_diff, 'Color', [0.8 0.4 0.0], 'LineWidth', 1.0);
            ylabel('$\|\mathbf{s}^{\mathrm{raw}} - \mathbf{s}^{\mathrm{ND}}\|$ (px)', ...
                   'Interpreter', 'latex', 'FontSize', 11);
            grid on; set(gca, 'FontSize', 10);
            text(0.02, 0.85, '(b) Feature observation difference', 'Units', 'normalized', ...
                 'Interpreter', 'latex', 'FontSize', 10);

            % (c) Error difference: |e_raw - e_nd|
            ee_diff = vecnorm(e_raw - e_nd, 2, 1);
            subplot(4, 1, 3);
            plot(t, ee_diff, 'Color', [0.6 0.0 0.6], 'LineWidth', 1.0);
            ylabel('$\|\mathbf{e}^{\mathrm{raw}} - \mathbf{e}^{\mathrm{ND}}\|$ (px)', ...
                   'Interpreter', 'latex', 'FontSize', 11);
            grid on; set(gca, 'FontSize', 10);
            text(0.02, 0.85, '(c) Feature error difference', 'Units', 'normalized', ...
                 'Interpreter', 'latex', 'FontSize', 10);

            % (d) Velocity difference: |v_raw - v_nd|
            vv_diff = vecnorm(v_raw - v_nd, 2, 1);
            subplot(4, 1, 4);
            plot(t, vv_diff, 'Color', [0.0 0.0 0.8], 'LineWidth', 1.0);
            ylabel('$\|\mathbf{v}^{\mathrm{raw}} - \mathbf{v}^{\mathrm{ND}}\|$', ...
                   'Interpreter', 'latex', 'FontSize', 11);
            xlabel('Control step $k$', 'Interpreter', 'latex', 'FontSize', 11);
            grid on; set(gca, 'FontSize', 10);
            text(0.02, 0.85, '(d) Velocity command difference', 'Units', 'normalized', ...
                 'Interpreter', 'latex', 'FontSize', 10);

            % ====================================================================
            %  Quantitative Metrics (Console Output)
            % ====================================================================
            fprintf('\n========== Velocity Comparison: Quantitative Metrics ==========\n');

            % 1. Velocity smoothness: std of step-to-step velocity change
            dv_nd = diff(v_nd, 1, 2);
            v_jitter_raw = std(vecnorm(dv_raw, 2, 1));
            v_jitter_nd  = std(vecnorm(dv_nd,  2, 1));
            fprintf('\n--- Velocity Smoothness ---\n');
            fprintf('  Velocity jitter (std of ||dv||):  Baseline = %.6f,  PF-NDRG = %.6f\n', ...
                    v_jitter_raw, v_jitter_nd);
            fprintf('  Both values are small => velocity is NOT the jitter source.\n');

            % 2. Displacement smoothness: std of step-to-step displacement change
            dd_nd = diff(d_nd, 1, 2);
            d_jitter_raw = std(vecnorm(dd_raw, 2, 1));
            d_jitter_nd  = std(vecnorm(dd_nd,  2, 1));
            fprintf('\n--- Displacement Smoothness ---\n');
            fprintf('  Displacement jitter (std of ||dd||):  Baseline = %.6f,  PF-NDRG = %.6f\n', ...
                    d_jitter_raw, d_jitter_nd);
            fprintf('  Jitter reduction: %.1f%%\n', (1 - d_jitter_nd / d_jitter_raw) * 100);

            % 3. Convergence rate comparison
            en_raw = vecnorm(e_raw, 2, 1);
            en_nd  = vecnorm(e_nd,  2, 1);
            % Find convergence step (error < threshold)
            thresh = 0.5;
            idx_raw = find(en_raw < thresh, 1, 'first');
            idx_nd  = find(en_nd  < thresh, 1, 'first');
            if isempty(idx_raw), idx_raw = n; end
            if isempty(idx_nd),  idx_nd  = n; end
            fprintf('\n--- Convergence Rate ---\n');
            fprintf('  Convergence step (||e|| < %.1f px):  Baseline = %d,  PF-NDRG = %d\n', ...
                    thresh, idx_raw, idx_nd);
            fprintf('  Overhead: %.1f%%\n', (idx_nd - idx_raw) / idx_raw * 100);

            % 4. Peak displacement comparison
            d_peak_raw = max(vecnorm(d_raw, Inf, 1));
            d_peak_nd  = max(vecnorm(d_nd,  Inf, 1));
            fprintf('\n--- Peak Displacement ---\n');
            fprintf('  ||Dp||_max (inf-norm):  Baseline = %.6f m,  PF-NDRG = %.6f m\n', ...
                    d_peak_raw, d_peak_nd);

            % 5. Displacement std (jitter metric for Table 2)
            d_std_raw = std(vecnorm(d_raw, 2, 1));
            d_std_nd  = std(vecnorm(d_nd,  2, 1));
            fprintf('\n--- Displacement Std (sigma_Dp) ---\n');
            fprintf('  sigma_Dp:  Baseline = %.6f m,  PF-NDRG = %.6f m\n', d_std_raw, d_std_nd);
            fprintf('  Jitter reduction: %.1f%%\n', (1 - d_std_nd / d_std_raw) * 100);

            % 6. Indirect feedback quantification
            fprintf('\n--- Indirect Feedback Chain ---\n');
            fprintf('  Max displacement difference:  %.6f m\n', max(dp_diff));
            fprintf('  Max feature difference:       %.2f px\n', max(fs_diff));
            fprintf('  Max error difference:         %.2f px\n', max(ee_diff));
            fprintf('  Max velocity difference:      %.6f m/s\n', max(vv_diff));
            if max(dp_diff) > 0 && max(fs_diff) > 0 && max(ee_diff) > 0
                fprintf('  Feedback amplification (max):  Dp->s = %.1fx,  s->e = %.2fx,  e->v = %.4fx\n', ...
                        max(fs_diff) / max(dp_diff), ...
                        max(ee_diff) / max(fs_diff), ...
                        max(vv_diff) / max(ee_diff));
            end

            % 7. Summary table for paper (Table 2 values)
            fprintf('\n========== Values for Table 2 (Section_Experiment.tex) ==========\n');
            fprintf('  ||Dp||_max (m):       Baseline = %.6f,  PF-NDRG = %.6f\n', d_peak_raw, d_peak_nd);
            fprintf('  sigma_Dp (m):         Baseline = %.6f,  PF-NDRG = %.6f\n', d_std_raw, d_std_nd);
            fprintf('  Jitter reduction (%%): %.1f%%\n', (1 - d_std_nd / d_std_raw) * 100);
            fprintf('  Convergence steps:    Baseline = %d,  PF-NDRG = %d\n', idx_raw, idx_nd);
            fprintf('  Overhead (%%):         %.1f%%\n', (idx_nd - idx_raw) / idx_raw * 100);
            fprintf('===============================================================\n\n');
        end


        function [uv_out, info] = legacy_fov_virtual_uv(vs, uv)
            % Cheng/RVC legacy virtual-FOV correction, kept on its own path.
            uv_out = uv;
            virtual_uv_base = uv;
            imsz = vs.camera.npix(:);
            W = imsz(1);
            H = imsz(2);
            margin = vs.legacy_fov_margin;
            if isempty(margin)
                margin = 100;
            end
            margin = max(0, margin);

            info = struct( ...
                'active', false, 'index', 0, 'shift', [0; 0], ...
                'real_min_boundary', vs.min_feature_boundary(uv), ...
                'virtual_min_boundary', vs.min_feature_boundary(uv));

            if vs.is_first_cal == 1
                vs.last_uv = uv;
                return;
            end

            index = 0;
            fov_num = 0;
            for i = 1:size(uv, 1)
                if i == 1
                    limit_high = W - margin;
                else
                    limit_high = H - margin;
                end
                for j = 1:size(uv, 2)
                    if uv(i,j) <= margin || uv(i,j) >= limit_high
                        fov_num = fov_num + 1;
                        index = j;
                    end
                end
            end

            if fov_num == 0
                vs.is_FOV = 0;
                return;
            end

            vs.is_FOV = 1;
            if isempty(vs.last_uv) || ~isequal(size(vs.last_uv), size(uv))
                vs.last_uv = uv;
                return;
            end

            % Match Cheng's reference implementation: skip the duplicate first
            % result after initialization, otherwise use last_uv as point_a.
            if all(uv(:) == vs.last_uv(:))
                return;
            end

            point_a = vs.last_uv(:, index);
            vs.last_uv = uv;
            point_b = uv(:, index);

            point_c = 2 * point_b - point_a;
            radio = norm(point_b - point_a);
            if radio < eps
                return;
            end

            point_s = vs.uv_star(:, index);
            target_vec = point_s - point_b;
            target_norm = norm(target_vec);
            if target_norm < eps
                return;
            end
            point_d = point_b + radio * target_vec / target_norm;

            point_e = (point_d + point_c) / 2;
            blend_vec = point_e - point_b;
            blend_norm = norm(blend_vec);
            if blend_norm < eps
                return;
            end
            point_f = point_b + radio * blend_vec / blend_norm;

            shift = point_f - point_b;
            uv_out(1,:) = virtual_uv_base(1,:) + shift(1);
            uv_out(2,:) = virtual_uv_base(2,:) + shift(2);

            virtual_change = any(uv_out(1,:) <= 0 | uv_out(1,:) >= W | ...
                uv_out(2,:) <= 0 | uv_out(2,:) >= H);
            iter = 0;
            max_iter = 50;
            while virtual_change && iter < max_iter
                delta = point_d - point_b;
                uv_out(1,:) = uv_out(1,:) + delta(1);
                uv_out(2,:) = uv_out(2,:) + delta(2);
                shift = shift + delta;

                fov_num = sum(uv_out(1,:) < 0 | uv_out(1,:) > W | ...
                    uv_out(2,:) < 0 | uv_out(2,:) > H);
                if fov_num > 0
                    uv_out(1,:) = uv_out(1,:) + delta(1);
                    uv_out(2,:) = uv_out(2,:) + delta(2);
                    shift = shift + delta;
                else
                    virtual_change = false;
                end
                iter = iter + 1;
            end

            info.active = true;
            info.index = index;
            info.shift = shift;
            info.virtual_min_boundary = vs.min_feature_boundary(uv_out);
        end

        function is_out = has_feature_outside_image(vs, uv)
            imsz = vs.camera.npix(:);
            is_out = any(uv(1,:) <= 0 | uv(1,:) >= imsz(1) | ...
                uv(2,:) <= 0 | uv(2,:) >= imsz(2));
        end

        function d = min_feature_boundary(vs, uv)
            imsz = vs.camera.npix(:);
            d = min([uv(1,:), uv(2,:), imsz(1) - uv(1,:), imsz(2) - uv(2,:)]);
        end

        function [Tz, wz, info] = partitioned_apf_z_control(vs, uv, uv_star)
            % Corke-Hutchinson partitioned IBVS:
            % rz=[Tz wz] is controlled by image area and longest-line angle.
            if ~isempty(vs.paper_theta_pair)
                idx_i = vs.paper_theta_pair(1);
                idx_j = vs.paper_theta_pair(2);
                line_theta = atan2(uv(2,idx_j) - uv(2,idx_i), uv(1,idx_j) - uv(1,idx_i));
            else
                [idx_i, idx_j, line_theta] = vs.partitioned_longest_segment(uv);
            end
            theta_star = atan2(uv_star(2,idx_j) - uv_star(2,idx_i), uv_star(1,idx_j) - uv_star(1,idx_i));
            theta_err = atan2(sin(theta_star - line_theta), cos(theta_star - line_theta));
            % The toolbox camera-pose convention makes positive omega_z move
            % the image angle in the opposite direction to the paper's sign.
            wz = -vs.paper_theta_gain * theta_err;
            wz = min(max(wz, -vs.paper_wz_max), vs.paper_wz_max);

            sigma = vs.partitioned_polygon_sigma(uv);
            sigma_star = vs.partitioned_polygon_sigma(uv_star);
            [F, rho_min] = vs.partitioned_boundary_repulsion(uv);

            Tz_demand = vs.paper_sigma_gain * (sigma_star - sigma) - F;
            Tz_unclipped = vs.paper_mu * vs.paper_tz_state + (1 - vs.paper_mu) * Tz_demand;
            vs.paper_tz_state = Tz_unclipped;
            Tz = min(max(Tz_unclipped, vs.paper_tz_min), vs.paper_tz_max);

            info.theta = line_theta;
            info.theta_star = theta_star;
            info.theta_err = theta_err;
            info.sigma = sigma;
            info.sigma_star = sigma_star;
            info.repulse = F;
            info.rho_min = rho_min;
            info.Tz_unclipped = Tz_unclipped;
            info.Tz = Tz;
            info.wz = wz;
            info.longest_pair = [idx_i idx_j];
        end

        function sigma = partitioned_polygon_sigma(~, uv)
            sigma = sqrt(max(polyarea(uv(1,:), uv(2,:)), 0));
        end

        function [idx_i, idx_j, line_theta] = partitioned_longest_segment(~, uv)
            n = size(uv, 2);
            idx_i = 1;
            idx_j = min(2, n);
            best_d2 = -inf;
            for a = 1:n-1
                for b = a+1:n
                    d = uv(:,b) - uv(:,a);
                    d2 = d' * d;
                    if d2 > best_d2
                        best_d2 = d2;
                        idx_i = a;
                        idx_j = b;
                    end
                end
            end
            line_theta = atan2(uv(2,idx_j) - uv(2,idx_i), uv(1,idx_j) - uv(1,idx_i));
        end

        function [F, rho_min] = partitioned_boundary_repulsion(vs, uv)
            imsz = vs.camera.npix(:);
            u = uv(1,:);
            v = uv(2,:);
            rho = min([u; v; imsz(1) - u; imsz(2) - v], [], 1);
            rho = max(rho, 1e-6);
            rho_min = min(rho);
            active = rho <= vs.paper_rho0;
            if any(active)
                % logarithmic boundary repulsion, zero at rho = rho0
                Fa = vs.paper_repulse_gain * log(vs.paper_rho0 ./ rho(active));
                F = max([0 Fa]);
            else
                F = 0;
            end
        end

        function [tau, s_ref_pred, exitflag] = mpcqp_controller_pixel(vs, s, s_target, B_jacobian, dt)
            s = s(:);
            s_target = s_target(:);
            n = numel(s);
            Np = max(1, round(vs.mpc_horizon));
            A = eye(n);
            B = dt * B_jacobian;
            tau_seed = vs.mpcqp_apply_velocity_limits(vs.mpc_tau_prev(:));
            tau_seq_seed = repmat(tau_seed, Np, 1);

            [H, f] = vs.mpcqp_generate_cost(s, s_target, A, B, Np);
            [A_vel, b_vel] = vs.mpcqp_velocity_constraints(Np, vs.mpc_v_max, vs.mpc_w_max);
            [G_fov, h_fov] = vs.mpcqp_fov_constraints(s, A, B, Np);

            qp_options = optimoptions('quadprog', 'Display', 'off');
            H_soft = blkdiag(H, 2 * vs.mpc_slack_weight);
            f_soft = [f; 0];
            G_soft = [
                A_vel, zeros(size(A_vel, 1), 1);
                G_fov, -ones(size(G_fov, 1), 1);
                zeros(1, 6 * Np), -1
            ];
            h_soft = [b_vel; h_fov; 0];

            solve_repeats = 1;
            if ~isempty(vs.mpc_solve_repeats)
                solve_repeats = max(1, round(vs.mpc_solve_repeats));
            end

            z0 = [tau_seq_seed; 0];
            z_soft = [];
            exitflag = -1;
            for solve_idx = 1:solve_repeats %#ok<NASGU>
                [z_soft, ~, exitflag] = quadprog(H_soft, f_soft, G_soft, h_soft, [], [], [], [], z0, qp_options);
                if exitflag <= 0 || isempty(z_soft)
                    break;
                end
                z0 = z_soft;
            end

            if exitflag > 0 && ~isempty(z_soft)
                tau_seq = z_soft(1:6 * Np);
                tau = vs.mpcqp_apply_velocity_limits(tau_seq(1:6));
                s_ref_pred = s + B * tau;
            else
                tau = [];
                s_ref_pred = [];
            end
        end

        function [H, f] = mpcqp_generate_cost(vs, s, s_target, A, B, Np)
            n = size(s, 1);
            m = size(B, 2);
            M_tau = zeros(n * Np, m * Np);
            M_s = zeros(n * Np, n);
            for i = 1:Np
                M_s((i - 1) * n + 1:i * n, :) = A^i;
                for j = 1:i
                    M_tau((i - 1) * n + 1:i * n, (j - 1) * m + 1:j * m) = A^(i - j) * B;
                end
            end

            Q = vs.mpcqp_weight_matrix(vs.mpc_q_weight, n);
            R = vs.mpcqp_weight_matrix(vs.mpc_r_weight, m);
            Q_blk = kron(eye(Np), Q);
            R_blk = kron(eye(Np), R);
            S_target = repmat(s_target, Np, 1);

            H = 2 * (M_tau' * Q_blk * M_tau + R_blk);
            H = (H + H') / 2;
            H = H + 1e-9 * eye(size(H));
            f = 2 * (M_tau' * Q_blk * (M_s * s - S_target));
        end

        function [A_vel, b_vel] = mpcqp_velocity_constraints(~, Np, v_max, w_max)
            I6 = eye(6);
            A_vel = [kron(eye(Np), I6); kron(eye(Np), -I6)];
            b_step = [v_max * ones(3,1); w_max * ones(3,1)];
            b_vel = repmat([b_step; b_step], Np, 1);
        end

        function [G_fov, h_fov] = mpcqp_fov_constraints(vs, s0, A, B, Np)
            n_s = numel(s0);
            n_tau = size(B, 2);
            A_big = zeros(n_s * Np, n_s);
            B_big = zeros(n_s * Np, n_tau * Np);

            for i = 1:Np
                A_big((i - 1) * n_s + 1:i * n_s, :) = A^i;
                for j = 1:i
                    B_big((i - 1) * n_s + 1:i * n_s, (j - 1) * n_tau + 1:j * n_tau) = A^(i - j) * B;
                end
            end

            n_feat = n_s / 2;
            Cu = zeros(n_feat, n_s);
            Cv = zeros(n_feat, n_s);
            for i = 1:n_feat
                Cu(i, 2 * i - 1) = 1;
                Cv(i, 2 * i) = 1;
            end

            Cu_big = kron(eye(Np), Cu);
            Cv_big = kron(eye(Np), Cv);

            img_w = vs.camera.npix(1);
            img_h = vs.camera.npix(2);
            margin = max(0, vs.mpc_fov_margin);
            u_min = margin;
            u_max = img_w - margin;
            v_min = margin;
            v_max = img_h - margin;

            if u_min > u_max
                u_min = 0;
                u_max = img_w;
            end
            if v_min > v_max
                v_min = 0;
                v_max = img_h;
            end

            G_fov = [
                Cu_big * B_big;
               -Cu_big * B_big;
                Cv_big * B_big;
               -Cv_big * B_big
            ];

            offset_u = Cu_big * A_big * s0;
            offset_v = Cv_big * A_big * s0;

            h_fov = [
                repmat(u_max, n_feat * Np, 1) - offset_u;
               -repmat(u_min, n_feat * Np, 1) + offset_u;
                repmat(v_max, n_feat * Np, 1) - offset_v;
               -repmat(v_min, n_feat * Np, 1) + offset_v
            ];
        end

        function W = mpcqp_weight_matrix(~, weight, dim)
            if isscalar(weight)
                W = weight * eye(dim);
            else
                W = weight;
            end
        end

        function tau = mpcqp_apply_velocity_limits(vs, tau)
            tau = tau(:);
            tau(1:3) = min(max(tau(1:3), -vs.mpc_v_max), vs.mpc_v_max);
            tau(4:6) = min(max(tau(4:6), -vs.mpc_w_max), vs.mpc_w_max);
        end

        function [Zest,Ztrue] = depth_estimator(vs, uv)

            if isempty(vs.uv_p)
                Zest = [];
                Ztrue = [];
                return;
            end

            % compute Jacobian for unit depth, z=1
            J = vs.camera.visjac_p(uv, 1);
            Jv = J(:,1:3);  % velocity part, depends on 1/z
            Jw = J(:,4:6);  % rotational part, indepedent of 1/z

            % estimate image plane velocity
            uv_d =  uv(:) - vs.uv_p(:);

            % estimate coefficients for A (1/z) = B
            B = uv_d - Jw*vs.vel_p(4:6);
            A = Jv * vs.vel_p(1:3);

            AA = zeros(numcols(uv), numcols(uv)/2);
            for i=1:numcols(uv)
                AA(i*2-1:i*2,i) = A(i*2-1:i*2);
            end
            eta = AA\B;          % least squares solution

            eta2 = A(1:2) \ B(1:2);

            % first order smoothing
            vs.theta = (1-vs.smoothing) * 1./eta' + vs.smoothing * vs.theta;
            Zest = vs.theta;

            % true depth
            P_CT = inv(vs.Tcam) * vs.P;
            Ztrue = P_CT(3,:);

            if vs.verbose
                fprintf('depth %.4g, est depth %.4g, rls depth %.4g\n', ...
                    Ztrue, 1/eta, Zest);
            end
        end
    end % methods
end % class
