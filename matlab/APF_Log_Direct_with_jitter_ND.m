%% APF-IBVS baseline: logarithmic FOV repulsion injected into the IBVS camera velocity
% Same initial scene, random sampling intervals and camera model as C_IBVS_PF_NDRG.m.
% Figures: feature trajectories, signed u/v errors, six camera velocities.
% With G_t: the raw increment is shaped by the neurodynamic model (A=0.4, B=D=0.05, K=8).
clc; close all;

%% Project and toolbox paths
project_dir = fileparts(mfilename('fullpath'));
addpath(project_dir,'-begin');
tb_dirs = {'spatial-math','vision','common'};
for i_tb = 1:numel(tb_dirs)
    if exist(fullfile(project_dir,tb_dirs{i_tb}),'dir')
        addpath(fullfile(project_dir,tb_dirs{i_tb}),'-begin');
    end
end
if exist(fullfile(project_dir,'Camera'),'dir')
    addpath(genpath(fullfile(project_dir,'Camera')),'-begin');
end

%% APF-IBVS parameters (Section 4.1.1)
cfg.seed = 42;
cfg.max_steps = 3000;
cfg.fps = 33;
cfg.jitter_ms = 5;
cfg.random_dt = true;
cfg.depth = 1;
cfg.tolerance = 0.5;
cfg.mu = 50;
cfg.k_log = 0.01;
cfg.gain_mode = 'fixed';
cfg.lambda = 0.4;
cfg.max_delta = 0.05;

%% Original comparison scene
cam = CentralCamera('default');
W = cam.npix(1); H = cam.npix(2);
P = mkgrid(2,0.5);
p_goal = bsxfun(@plus,200*[-1 -1 1 1;-1 1 1 -1],cam.pp');
T = SE3(-2.1,0,-3.0)*SE3.Rz(5*pi/4);
uv0 = cam.project(P,'pose',T);
assert(all(isfinite(uv0(:))),'Invalid initial projection');
assert(min_fov_margin(uv0,W,H)>0,'Initial features outside FOV');
assert(min_fov_margin(p_goal,W,H)>0,'Desired features outside FOV');

rng(cfg.seed,'twister');
dt_ms = repmat(round(1000/cfg.fps),cfg.max_steps,1);
if cfg.random_dt
    dt_ms = max(1,dt_ms+randi([-cfg.jitter_ms cfg.jitter_ms],cfg.max_steps,1));
end

%% Closed loop: log force injected directly into the IBVS camera velocity
npts = size(p_goal,2);
Nmax = cfg.max_steps;
features = nan(numel(uv0),Nmax+1);
features(:,1) = uv0(:);
err_norm = nan(1,Nmax+1);
err_norm(1) = norm(uv0(:)-p_goal(:));
margin = nan(1,Nmax+1);
margin(1) = min_fov_margin(uv0,W,H);
exec_vel = nan(6,Nmax);
exec_delta = nan(6,Nmax); % executed 6-DOF pose increments
jcond = nan(1,Nmax);
dt_used = nan(1,Nmax);
status = 'max_steps';
N = 1;
nd_pos_step_state = zeros(6,1); % ND state, reset for this run

for k = 1:Nmax
    if err_norm(N)<=cfg.tolerance
        status = 'converged';
        break;
    end
    dt = dt_ms(k)/1000;
    uv = reshape(features(:,N),2,[]);
    Frep = log_fov_force(uv,W,H,cfg.mu,cfg.k_log);
    e = uv(:)-p_goal(:);
    L = cam.visjac_p(uv,cfg.depth);
    Jplus = pinv(L);
    cond_now = cond(L);
    if strcmp(cfg.gain_mode,'original1')
        q = norm(Jplus*e,inf);
        lambda = 1.1*exp(-(30/1.1)*q)+0.4;
    else
        lambda = cfg.lambda;
    end
    v = -lambda*(Jplus*e) + Jplus*Frep(:);
    % ND displacement-increment smoothing, copied from IBVS.m step().
    raw_v_delta = v * dt;
    K_gain = 8.0;
    A_step = 0.4;
    B_step = 0.05;
    D_step = 0.05;
    I_plus = max(0, raw_v_delta * K_gain);
    I_minus = max(0, -raw_v_delta * K_gain);
    Step_curr = nd_pos_step_state;
    dStep = -A_step * Step_curr + (B_step - Step_curr).*I_plus ...
        - (D_step + Step_curr).*I_minus;
    Step_next = Step_curr + dStep;
    Step_next = min(B_step, max(-D_step, Step_next));
    nd_pos_step_state = Step_next;
    dq = Step_next;
    if any(~isfinite(dq))
        status = 'invalid_command';
        break;
    end
    try
        Tnew = T*SE3(trnorm(delta2tr(dq)));
        uvnew = cam.project(P,'pose',Tnew);
    catch
        status = 'projection_error';
        break;
    end
    if any(~isfinite(uvnew(:)))
        status = 'invalid_projection';
        break;
    end
    exec_delta(:,N) = dq(:);
    exec_vel(:,N) = dq(:)/dt; % executed camera twist (m/s, rad/s)
    jcond(N) = cond_now;
    dt_used(N) = dt;
    N = N+1;
    features(:,N) = uvnew(:);
    err_norm(N) = norm(uvnew(:)-p_goal(:));
    margin(N) = min_fov_margin(uvnew,W,H);
    T = Tnew;
    if margin(N)<=0
        status = 'FOV_violation';
        break;
    end
    if err_norm(N)<=cfg.tolerance
        status = 'converged';
        break;
    end
end

features = features(:,1:N);
err_norm = err_norm(1:N);
margin = margin(1:N);
feature_error = features - p_goal(:);
exec_vel = exec_vel(:,1:N-1);
exec_delta = exec_delta(:,1:N-1);
jcond = jcond(1:N-1);
dt_used = dt_used(1:N-1);
point_error = zeros(npts,N);
for p = 1:npts
    point_error(p,:) = hypot(feature_error(2*p-1,:),feature_error(2*p,:));
end

fprintf('\nA: Log + Direct (single-method result)\n');
fprintf('Status: %s | steps: %d | final real-feature error: %.4f px | min FOV margin: %.3f px\n', ...
    status,N-1,err_norm(end),min(margin));

%% Figure 1: actual image-feature convergence trajectories (same A run)
figure('Name','A - Log + Direct: Image Feature Trajectories', ...
    'Color','w','Visible','on','Position',[80 90 900 720]);
ax_traj = axes('Position',[0.10 0.10 0.68 0.82]);
hold(ax_traj,'on'); grid(ax_traj,'on'); box(ax_traj,'on');
set(ax_traj,'YDir','reverse','FontName','Times New Roman','FontSize',12);
xlim(ax_traj,[0 W]); ylim(ax_traj,[0 H]);
xlabel(ax_traj,'u (pixels)'); ylabel(ax_traj,'v (pixels)');
title(ax_traj,sprintf('A: Log + Direct | %s | e=%.2f px', ...
    status,err_norm(end)),'Interpreter','none');
rectangle(ax_traj,'Position',[cfg.mu cfg.mu W-2*cfg.mu H-2*cfg.mu], ...
    'LineStyle','--','EdgeColor',[0.65 0.65 0.65],'LineWidth',1.0);
start_contour = [uv0 uv0(:,1)];
goal_contour = [p_goal p_goal(:,1)];
plot(ax_traj,start_contour(1,:),start_contour(2,:),'g--', ...
    'LineWidth',1.3,'DisplayName','Initial contour');
plot(ax_traj,goal_contour(1,:),goal_contour(2,:),'r:', ...
    'LineWidth',1.4,'DisplayName','Desired contour');
plot(ax_traj,p_goal(1,:),p_goal(2,:),'r*','MarkerSize',8, ...
    'DisplayName','Desired features');
traj_colors = lines(npts);
for p = 1:npts
    plot(ax_traj,features(2*p-1,:),features(2*p,:),'-', ...
        'Color',traj_colors(p,:),'LineWidth',1.5, ...
        'DisplayName',sprintf('Actual feature %d',p));
end
uvlast = reshape(features(:,end),2,[]);
plot(ax_traj,uv0(1,:),uv0(2,:),'go','MarkerFaceColor','g', ...
    'HandleVisibility','off');
plot(ax_traj,uvlast(1,:),uvlast(2,:),'ks', ...
    'MarkerFaceColor','k','DisplayName','Final features');
legend(ax_traj,'Location','bestoutside','Interpreter','none','FontSize',9);
drawnow;

%% Figure 2: eight signed feature-error components (reference: second script)
figure('Name','A - Signed Feature Errors','Color','w','Position',[110 110 1100 680]);
ax3 = axes('Position',[0.09 0.13 0.86 0.78]);
hold(ax3,'on'); grid(ax3,'on'); box(ax3,'on');
cc = lines(npts);
for p = 1:npts
    plot(ax3,0:N-1,feature_error(2*p-1,:),'-','Color',cc(p,:), ...
        'LineWidth',1.4,'DisplayName',sprintf('e_{u%d}',p));
    plot(ax3,0:N-1,feature_error(2*p,:),'--','Color',cc(p,:), ...
        'LineWidth',1.4,'DisplayName',sprintf('e_{v%d}',p));
end
xlabel(ax3,'Number of iterations'); ylabel(ax3,'Signed feature error (pixels)');
title(ax3,'A: Log + Direct - Feature Error Components');
set(ax3,'FontName','Times New Roman','FontSize',11);
legend(ax3,'Location','best','NumColumns',4);

%% Figure 3: six camera velocity components in ONE axes (reference style)
figure('Name','A - Camera Velocity','Color','w','Position',[120 100 900 600]);
axv = axes('Position',[0.11 0.13 0.84 0.78]);
hold(axv,'on'); grid(axv,'on'); box(axv,'on');
vel_labels = {'v_x','v_y','v_z','\omega_x','\omega_y','\omega_z'};
vel_colors = [0.000 0.447 0.741; 0.850 0.325 0.098; ...
              0.929 0.694 0.125; 0.494 0.184 0.556; ...
              0.466 0.674 0.188; 0.301 0.745 0.933];
vel_styles = {'-','-','-','--','--','--'};
for c = 1:6
    plot(axv,1:size(exec_vel,2),exec_vel(c,:),vel_styles{c}, ...
        'Color',vel_colors(c,:),'LineWidth',1.5, ...
        'DisplayName',vel_labels{c});
end
xlabel(axv,'Number of iterations');
ylabel(axv,'Cartesian velocity (m/s, rad/s)');
yline(axv,0,'k:','HandleVisibility','off');
set(axv,'FontName','Times New Roman','FontSize',12);
legend(axv,'Location','northeast','NumColumns',2,'Interpreter','tex','Box','on');
drawnow;

%% Six performance metrics, computed from the actual A-method run
metrics = struct();
metrics.steps = N-1;
metrics.final_feature_error_px = err_norm(end);
if isempty(jcond)
    metrics.mean_jacobian_condition = NaN;
else
    valid_cond = jcond(isfinite(jcond));
    if isempty(valid_cond)
        metrics.mean_jacobian_condition = NaN;
    else
        metrics.mean_jacobian_condition = mean(valid_cond);
    end
end
if isempty(exec_vel)
    metrics.max_camera_velocity_norm = NaN;
else
    metrics.max_camera_velocity_norm = max(sqrt(sum(exec_vel.^2,1)));
end
metrics.min_fov_margin_px = min(margin);
% Displacement-increment jitter: RMS of consecutive 6-DOF increment differences.
% This is an explicitly defined comparison metric; verify the paper's exact
% sigma definition and unit normalization before claiming numerical replication.
if size(exec_delta,2) < 2
    metrics.displacement_increment_jitter = NaN;
    delta_increment_diff = zeros(6,0);
else
    delta_increment_diff = diff(exec_delta,1,2);
    metrics.displacement_increment_jitter = ...
        sqrt(mean(sum(delta_increment_diff.^2,1)));
end
% Extra diagnostics, kept separate from the five headline metrics
metrics.fov_violations = sum(margin<=0);
metrics.converged = strcmp(status,'converged');
metrics.status = {status};
fprintf('\n=== Six performance metrics: A Log + Direct ===\n');
fprintf('Iterations: %d\n',metrics.steps);
fprintf('Final feature error: %.6f px\n',metrics.final_feature_error_px);
fprintf('Mean Jacobian condition number: %.6f\n',metrics.mean_jacobian_condition);
fprintf('Max camera twist norm: %.6f (mixed linear/angular units)\n',metrics.max_camera_velocity_norm);
fprintf('Minimum FOV margin: %.6f px\n',metrics.min_fov_margin_px);
fprintf('Displacement increment jitter (RMS): %.9f (mixed m/rad)\n', ...
    metrics.displacement_increment_jitter);

%% Local functions: logarithmic FOV repulsion
function f = log_fov_force(uv,W,H,mu,kr)
    f = zeros(size(uv));
    for p = 1:size(uv,2)
        x = uv(1,p); y = uv(2,p);
        if x<mu
            f(1,p) = f(1,p)+axis_repulsion(x,W,mu,kr);
        end
        if x>W-mu
            f(1,p) = f(1,p)-axis_repulsion(W-x,W,mu,kr);
        end
        if y<mu
            f(2,p) = f(2,p)+axis_repulsion(y,H,mu,kr);
        end
        if y>H-mu
            f(2,p) = f(2,p)-axis_repulsion(H-y,H,mu,kr);
        end
    end
end

function f = axis_repulsion(d,extent,mu,kr)
    d = max(1e-6,d);
    x = d; U = extent;
    A = log(max((U-x)/mu,realmin));
    B = log(max(mu/x,realmin));
    xi = (x-U/2)^2;
    dxi = 2*(x-U/2);
    dA = -1/max(U-x,1e-6);
    dB = -1/x;
    dV = dxi*A*B+xi*(dA*B+A*dB);
    f = -kr*dV;
end

function d = min_fov_margin(uv,W,H)
    dist = [uv(1,:);uv(2,:);W-uv(1,:);H-uv(2,:)];
    d = min(dist(:));
end
