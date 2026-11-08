%VisualServo  Abstract class for visual servoing
%
% VisualServo(CAMERA, OPTIONS) 
%
% Methods::
% run            Run the simulation, complete results kept in the object
% step           One simulation step (provided by the concrete class)
% init           Initialized the simulation (provided by the concrete class)
% plot_p         Plot feature values vs time
% plot_vel       Plot camera velocity vs time
% plot_camera    Plot camera pose vs time
% plot_jcond     Plot Jacobian condition vs time 
% plot_z         Plot point depth vs time
% plot_error     Plot feature error vs time
% plot_all       Plot all of the above in separate figures
% char           Convert object to a concise string
% display        Display the object as a string
%
% Properties::
% history     A vector of structs holding simulation results
%
% Notes::
% - Must be subclassed.
%
% See also PBVS, IBVS, IBVS_l, IBVS_e.

classdef VisualServo < handle
    properties
        P
        uv_star
        Tcam
        camera         % camera object
        Tf
        T0
        pf

        history         % vector of structs to hold EKF history
        history_low_level %底层运动控制器搜集器
        
        niter
        fps

        verbose
        arglist
        axis
        
        movie
        anim
        
        %下面的变量都是自己添加的
        type
        fis %fuzzy logic object
        e_last%Error of the last iteration cycle
        gain_type

        %初始速度连续性相关代码
        first_iteration %记录是否第一次迭代计算的标志位
        first_e %初始迭代时图像雅可比和特征误差的乘积
        first_state1;
        first_state2;
        first_state3;
        first_state4;
        first_state5;
        first_state6;
        time %计时器
        mu %初始连续因子

        %下述6个变量用于获取六个维度状态空间的最值
        max_s1
        max_s2
        max_s3
        max_s4
        max_s5
        max_s6
        %强化学习相关参数
        state_num%状态空间数量
        action_num%动作空间数量
        train_num%训练轮数
        alapha%学习因子
        gamma%折扣因子
        beta%动作选择概率
        reward%奖励数值，是一个常量
        %状态空间矩阵,即Q表
        q_table_1
        q_table_2
        q_table_3
        q_table_4
        q_table_5
        q_table_6
        state_discrete%根据状态空间最值划分得到离散的状态空间变量，这是一个6x20的矩阵，行数代表维度，列数代表状态数量
        action_discrete%动作空间离散变量可能的取值，是一个1x9的增益数值向量
        is_train%标志位，当为1时，开始强化学习训练，当为0时，结束强化学习的训练
        %关于FOV约束的相关参数
        last_uv ;%存储上个迭代周期的像素坐标信息
        is_FOV ;%判断像素坐标是否满足FOV约束
        is_first_cal ;%是否为第一次计算迭代
        %随机采样周期（任务处理周期）相关参数
        Task_process_cycle_min ; %任务处理周期波动最小值
        Task_process_cycle_max ;%任务处理周期波动最大值
        Task_process_clcle ;%任务处理周期
        delta_x ;%x方向插补加速度
        delta_y; %y方向插补加速度
        delta_z; %z方向插补加速度
        delta_a; %a方向插补加速度
        delta_b; %b方向插补加速度
        delta_c; %c方向插补加速度
        a_x_max;%x方向最大加速度
        a_y_max;%y方向最大加速度
        a_z_max;%z方向最大加速度
        a_a_max;%a方向最大加速度
        a_b_max;%b方向最大加速度
        a_c_max;%c方向最大加速度
        v_base; %每次插补的前一时刻的速度
        count_slot_positive;%统计伺服任务出现正时隙的次数
        count_slot_negative;%统计伺服任务出现负时隙的次数
        count_slot_zero;%统计伺服任务出现零时隙的次数
        count_high_level;%统计伺服全过程high lelvel下发执行的次数
        %设计观测器时所需的参数变量
        fix_clycle;%假设观测器每fix_cycle周期就向底层发送一次期望速度
        last_e;%存储上个周期的伺服误差
        beta_jacobian;%图像雅可比的矩阵增益系数
        fis_jacobian;%矩阵增益系数的fuzzy object
        error_bate_last;
        desired_depth;

    end

    methods

        function vs = VisualServo(cam, varargin)
            %VisualServo.VisualServo Create IBVS object
            %
            % VS = VisualServo(camera, options) creates an image-based visual servo
            % simulation object.
            %
            % Options::
            % 'niter',N         Maximum number of iterations
            % 'fps',F           Number of simulation frames per second (default t)
            % 'Tf',T            The final pose
            % 'T0',T            The initial pose
            % 'P',p             The set of world points (3xN)
            % 'targetsize',S    The target points are the corners of an SxS
            %                   square in the XY-plane at Z=0 (default S=0.5)
            % 'pstar',p         The desired image plane coordinates
            % 'verbose'         Print out extra information during simulation
            %
            % Notes::
            % - If 'P' is specified it overrides the default square target.

            
            vs.camera = cam;
            vs.history = [];
            vs.history_low_level = [];

            z = 3;
            opt.niter = [];
            opt.fps = 33;
            opt.posef = [];
            opt.pose0 = cam.T;
            opt.P = [];
            opt.targetsize = 0.5;       % dimensions of target
            opt.pstar = [];
            opt.axis = [];
            opt.movie = [];
            opt.gain_type = [];
            
            [opt,vs.arglist] = tb_optparse(opt, varargin);
            vs.niter = opt.niter;
            vs.fps = opt.fps;
            vs.verbose = opt.verbose;
            vs.e_last = 0;
            vs.gain_type = opt.gain_type;

            vs.first_iteration = 1;
            vs.first_e = 0;
            vs.first_state1 = 0;
            vs.first_state2 = 0;
            vs.first_state3 = 0;
            vs.first_state4 = 0;
            vs.first_state5 = 0;
            vs.first_state6 = 0;
            vs.time = 0;
            vs.mu = 4;
            %FOV约束相关的初始化设置
            vs.last_uv = 0;
            vs.is_FOV = 0;
            vs.is_first_cal = 1;
            %在这里添加fuzzy logic相关代码
            if opt.gain_type == 2
                %在这里添加关于fuzzylogic的代码
                fis = mamfis('Name','Gain');
                fis = addInput(fis,[0 1],'Name','error');
                fis = addMF(fis,'error','gbellmf',[0.2 1.5 0],'Name','L');
                fis = addMF(fis,'error','gbellmf',[0.2 2 0.5],'Name','M');
                fis = addMF(fis,'error','gbellmf',[0.2 1.5 1],'Name','H');
                
                fis = addInput(fis,[-1 1],'Name','error_change_rate');
                fis = addMF(fis,'error_change_rate','gbellmf',[0.4166 2.5 -1],'Name','L');
                fis = addMF(fis,'error_change_rate','gbellmf',[0.4166 2.5 0],'Name','M');
                fis = addMF(fis,'error_change_rate','gbellmf',[0.4166 2.5 1],'Name','H');
                
                fis = addOutput(fis,[0 2],'Name','lambda');
                fis = addMF(fis,'lambda','gaussmf',[0.3538 0],'Name','L');
                fis = addMF(fis,'lambda','gaussmf',[0.3538 1],'Name','M');
                fis = addMF(fis,'lambda','gaussmf',[0.3538 2],'Name','H');
                
                ruleList = [1 1 3 1 1
                    1 2 3 1 1
                    1 3 2 1 1
                    2 1 2 1 1
                    2 2 2 1 1
                    2 3 1 1 1
                    3 1 2 1 1
                    3 2 2 1 1
                    3 3 1 1 1];
                fis = addRule(fis,ruleList);
                vs.fis = fis;
            end
            
            %关于强化学习的相关参数配置
            if opt.gain_type == 0
                %这一部分主要是针对C-IBVS的单次运行过程去过去max_s所进行的配置
                vs.max_s1 = 0;
                vs.max_s2 = 0;
                vs.max_s3 = 0;
                vs.max_s4 = 0;
                vs.max_s5 = 0;
                vs.max_s6 = 0;
            end
            if opt.gain_type == 3
                %配置强化学习的相关参数
                vs.state_num = 20;
                vs.action_num = 9;
                vs.train_num = 100;
                vs.alapha = 0.5;
                vs.gamma = 0.85;
                vs.beta = 0.9;
                vs.reward = 100;
                %创建状态空间矩阵
                vs.q_table_1 = zeros(vs.state_num,vs.action_num);
                vs.q_table_2 = zeros(vs.state_num,vs.action_num);
                vs.q_table_3 = zeros(vs.state_num,vs.action_num);
                vs.q_table_4 = zeros(vs.state_num,vs.action_num);
                vs.q_table_5 = zeros(vs.state_num,vs.action_num);
                vs.q_table_6 = zeros(vs.state_num,vs.action_num);
                %接下来初始化状态空间最值范围，为状态空间的离散化做准备
                vs.max_s1 = 1;
                vs.max_s2 = 1;
                vs.max_s3 = -1.48;
                vs.max_s4 = -1.74;
                vs.max_s5 = -3.17;
                vs.max_s6 = 1.70;
                %开始对状态空间进行离散化处理
                for i = 1 : 6
                    for j=1 : vs.state_num
                        if i ==1
                            vs.state_discrete (i,j) = vs.max_s1 * ( 1 - ( log(vs.state_num - j + 1) / log(vs.state_num + 1) ) ); 
                        end
                        if i ==2
                            vs.state_discrete (i,j) = vs.max_s2 * ( 1 - ( log(vs.state_num - j + 1) / log(vs.state_num + 1) ) ); 
                        end
                        if i ==3
                            vs.state_discrete (i,j) = vs.max_s3 * ( 1 - ( log(vs.state_num - j + 1) / log(vs.state_num + 1) ) ); 
                        end
                        if i ==4
                            vs.state_discrete (i,j) = vs.max_s4 * ( 1 - ( log(vs.state_num - j + 1) / log(vs.state_num + 1) ) ); 
                        end
                        if i ==5
                            vs.state_discrete (i,j) = vs.max_s5 * ( 1 - ( log(vs.state_num - j + 1) / log(vs.state_num + 1) ) ); 
                        end
                        if i ==6
                            vs.state_discrete (i,j) = vs.max_s6 * ( 1 - ( log(vs.state_num - j + 1) / log(vs.state_num + 1) ) ); 
                        end
                    end 
                end 
                %创建动作空间离散变量结构
                vs.action_discrete = 0.2 : 0.1 : 1;
                %初始化强化学习标志位
                vs.is_train = 1;
            end 
            %=====并行处理方法相关参数配置
            vs.Task_process_cycle_min = 0;
            vs.Task_process_cycle_max = 20;
            vs.Task_process_clcle = 1 / vs.fps * 1000;
            vs.delta_x = 0.0001;
            vs.delta_y = 0.0001;
            vs.delta_z = 0.0001;
            vs.delta_a = 0.0001;
            vs.delta_b = 0.0001;
            vs.delta_c = 0.0001;
            vs.a_x_max = 0.0005;
            vs.a_y_max = 0.0005;
            vs.a_z_max = 0.0005;
            vs.a_a_max = 0.0005;
            vs.a_b_max = 0.0005;
            vs.a_c_max = 0.0005;
            vs.v_base = zeros(6,1);
            vs.count_slot_positive = 0;
            vs.count_slot_negative = 0;
            vs.count_slot_zero = 0;
            vs.count_high_level = 0;
            %设计观测器所需的参数变量
            vs.fix_clycle = 5;
            %对图像雅可比增益系数进行初始化设置,并且进行模糊规则的定义
            vs.beta_jacobian = 0;
            %在这里添加关于fuzzylogic的代码
            fis_jacobian = mamfis('Name','Gain_Jacobian');
            fis_jacobian = addInput(fis_jacobian,[0 1],'Name','error');
            fis_jacobian = addMF(fis_jacobian,'error','gbellmf',[0.2 1.5 0],'Name','L');
            fis_jacobian = addMF(fis_jacobian,'error','gbellmf',[0.2 2 0.5],'Name','M');
            fis_jacobian = addMF(fis_jacobian,'error','gbellmf',[0.2 1.5 1],'Name','H');

            fis_jacobian = addInput(fis_jacobian,[-1 1],'Name','error_change_rate');
            fis_jacobian = addMF(fis_jacobian,'error_change_rate','gbellmf',[0.4166 2.5 -1],'Name','L');
            fis_jacobian = addMF(fis_jacobian,'error_change_rate','gbellmf',[0.4166 2.5 0],'Name','M');
            fis_jacobian = addMF(fis_jacobian,'error_change_rate','gbellmf',[0.4166 2.5 1],'Name','H');

            fis_jacobian = addOutput(fis_jacobian,[0 1],'Name','lambda');
            fis_jacobian = addMF(fis_jacobian,'lambda','gaussmf',[0.22 0],'Name','L');
            fis_jacobian = addMF(fis_jacobian,'lambda','gaussmf',[0.22 0.5],'Name','M');
            fis_jacobian = addMF(fis_jacobian,'lambda','gaussmf',[0.22 1],'Name','H');

            ruleList = [1 1 1 1 1
                1 2 2 1 1
                1 3 2 1 1
                2 1 2 1 1
                2 2 3 1 1
                2 3 3 1 1
                3 1 3 1 1
                3 2 3 1 1
                3 3 3 1 1];
            fis_jacobian = addRule(fis_jacobian,ruleList);
            vs.fis_jacobian = fis_jacobian;
            vs.error_bate_last = 0;
            vs.desired_depth =  0;

            if ~isempty(opt.pose0)
                vs.T0 = SE3(opt.pose0);
            end
            if ~isempty(opt.posef)
                vs.Tf = SE3(opt.posef);
            end

            % define feature points in XY plane, make vertices of a square
            if isempty(opt.P)
                opt.P = mkgrid(2, opt.targetsize);
            end
            vs.P = opt.P;
            vs.pf = opt.pstar;
            vs.axis = opt.axis;
            vs.movie = opt.movie;
        end

        function run(vs, nsteps)
            %VisualServo.run  Run visual servo simulation
            %
            % VS.run(N) run the simulation for N steps
            %
            % VS.run() as above but run it for the number of steps
            % specified in the constructor or Inf.
            %
            % Notes::
            % - Repeatedly calls the subclass step() method which returns
            %   a flag to indicate if the simulation is complete.
            
            vs.init();
            
                        if ~isempty(vs.movie)
                vs.anim = Animate(vs.movie);
            end
            
            if nargin < 2
                nsteps = vs.niter;
            end
            ksteps = 0;
            while true
                ksteps = ksteps + 1;
%                 vs.count_high_level = vs.count_high_level + 1;
                %每次step迭代之前，生成当前任务处理周期
                range = (vs.Task_process_cycle_max - vs.Task_process_cycle_min) * rand(1,1) + vs.Task_process_cycle_min ;
                vs.Task_process_clcle = round(1 / vs.fps * 1000 +  range);
                status = vs.step();

                drawnow
                
                            if ~isempty(vs.movie)
                                vs.anim.add();
                            else
                                % pause(1/vs.fps)
                                pause(1/33000)   % simulate real time，默认和33fps匹配，可根据需要修改
                            end
                
                if status > 0
                    fprintf('completed on error tolerance\n');
                    break;
                elseif status < 0
                    fprintf('failed on error\n');
                    break;
                end
                
                if ~isempty(nsteps) && (ksteps > nsteps)
                    break;
                end
            end
            
            if ~isempty(vs.movie)
                vs.anim.close();
            end

            if status == 0
                fprintf('completed on iteration count\n');
            end
        end

        function plot_p(vs)
            %VisualServo.plot_p Plot feature trajectory
            %
            % VS.plot_p() plots point feature trajectories on the image plane.
            %
            % See also VS.plot_vel, VS.plot_error, VS.plot_camera,
            % VS.plot_jcond, VS.plot_z, VS.plot_error, VS.plot_all.
            
            if isempty(vs.history)
                return
            end
            if strcmp(vs.type, 'point') == 0
                disp('Can only plot image plane trajectories for point-based IBVS');
                return
            end
            clf
            hold on
            
            % image plane trajectory
            vs_type=class(vs);
            switch vs_type
                case 'PBVS'
                    uv = [vs.history.uv]';
                case 'IBVS'
                    uv = [vs.history.f]';
                otherwise
                    error('no vs type!');
            end
            % result is a vector with row per time step, each row is u1, v1, u2, v2 ...
            for i=1:numcols(uv)/2
                p = uv(:,i*2-1:i*2);    % get data for i'th point
                plot(p(:,1), p(:,2), 'b')
            end
            
            % mark the initial target shape
            plot_poly( reshape(uv(1,:), 2, []), 'o--');
            uv(end,:)
            
            % mark the final target shape
            if ~isempty(vs.uv_star)
                plot_poly(vs.uv_star, 'rh:', 'MarkerSize', 8, 'MarkerFaceColor', 'r')
            else
                plot_poly( reshape(uv(end,:), 2, []), 'rh--', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
            end
            axis([0 vs.camera.npix(1) 0 vs.camera.npix(2)]);
            daspect([1 1 1])
            set(gca, 'Ydir' , 'reverse');
            grid
            xlabel('u (pixels)');
            ylabel('v (pixels)');
            
            hold off
        end

       function plot_vel(vs)      % 20251228修改，注释掉原有的plot_vel函数
            %VisualServo.plot_vel Plot camera trajectory
            %
            % VS.plot_vel() plots the camera velocity versus time.
            %
            % See also VS.plot_p, VS.plot_error, VS.plot_camera,
            % VS.plot_jcond, VS.plot_z, VS.plot_error, VS.plot_all.
            if isempty(vs.history)
                return
            end
            clf
            vel = [vs.history.vel]';
            plot(vel(:,1:3), '-')
            hold on
            plot(vel(:,4:6), '--')
            hold off
            ylabel('Cartesian velocity')
            grid
            xlabel('Time step')
            xaxis(length(vs.history));
            legend('v_x', 'v_y', 'v_z', '\omega_x', '\omega_y', '\omega_z')
        end
        
        % function plot_vel(vs)   % 20251228修改，注释掉原有的plot_vel函数
        %     if isempty(vs.history), return; end

        %     % create / reuse named figure
        %     h = findobj('Type','figure','Name','Camera velocities');
        %     if isempty(h)
        %         h = figure('Name','Camera velocities','NumberTitle','off');
        %     else
        %         figure(h); clf;
        %     end

        %     vel = [vs.history.vel]';  % Nx6
        %     n = size(vel,1);
        %     colors = lines(6);
        %     lw = 1.5;

        %     hold on
        %     plot(1:n, vel(:,1), 'Color', colors(1,:), 'LineWidth', lw);    % v_x
        %     plot(1:n, vel(:,2), 'Color', colors(2,:), 'LineWidth', lw);    % v_y
        %     plot(1:n, vel(:,3), 'Color', colors(3,:), 'LineWidth', lw);    % v_z
        %     plot(1:n, vel(:,4), '--', 'Color', colors(4,:), 'LineWidth', lw); % omega_x
        %     plot(1:n, vel(:,5), '--', 'Color', colors(5,:), 'LineWidth', lw); % omega_y
        %     plot(1:n, vel(:,6), '--', 'Color', colors(6,:), 'LineWidth', lw); % omega_z
        %     hold off

        %     grid on
        %     xlabel('Time step')
        %     ylabel('Cartesian velocity')
        %     xaxis(length(vs.history));
        %     legend('v_x','v_y','v_z','\omega_x','\omega_y','\omega_z', 'Location', 'best');
        % end

        function plot_low_level_vel_predict(vs)
            %VisualServo.plot_vel Plot camera trajectory
            %
            % VS.plot_vel() plots the camera velocity versus time.
            %
            % See also VS.plot_p, VS.plot_error, VS.plot_camera,
            % VS.plot_jcond, VS.plot_z, VS.plot_error, VS.plot_all.
            if isempty(vs.history_low_level)
                return
            end
            clf
            low_level_vel_predict = [vs.history_low_level.low_level_vel_predict]';
            plot(low_level_vel_predict(:,1:3), '-')
            hold on
            plot(low_level_vel_predict(:,4:6), '--')
            hold off
            ylabel('Cartesian velocity')
            grid
            xlabel('Time step')
            xaxis(length(vs.history_low_level));
            legend('v_x', 'v_y', 'v_z', '\omega_x', '\omega_y', '\omega_z')
        end

        function plot_low_level_vel_desired(vs)
            %VisualServo.plot_vel Plot camera trajectory
            %
            % VS.plot_vel() plots the camera velocity versus time.
            %
            % See also VS.plot_p, VS.plot_error, VS.plot_camera,
            % VS.plot_jcond, VS.plot_z, VS.plot_error, VS.plot_all.
            if isempty(vs.history_low_level)
                return
            end
            clf
            low_level_vel_desired = [vs.history_low_level.low_level_vel_desired]';
            plot(low_level_vel_desired(:,1:3), '-')
            hold on
            plot(low_level_vel_desired(:,4:6), '--')
            hold off
            ylabel('desired Cartesian velocity')
            grid
            xlabel('Time step')
            xaxis(length(vs.history_low_level));
            legend('v_x', 'v_y', 'v_z', '\omega_x', '\omega_y', '\omega_z')
        end

        function plot_low_level_vel_predict_desired(vs)
            %VisualServo.plot_vel Plot camera trajectory
            %
            % VS.plot_vel() plots the camera velocity versus time.
            %
            % See also VS.plot_p, VS.plot_error, VS.plot_camera,
            % VS.plot_jcond, VS.plot_z, VS.plot_error, VS.plot_all.
            if isempty(vs.history_low_level)
                return
            end
            clf
            low_level_vel_predict = [vs.history_low_level.low_level_vel_predict]';
            plot(low_level_vel_predict(:,1:3), '-')
            hold on
            plot(low_level_vel_predict(:,4:6), '--')
            hold on
            low_level_vel_desired = [vs.history_low_level.low_level_vel_desired]';
            plot(low_level_vel_desired(:,1:3), '-')
            hold on
            plot(low_level_vel_desired(:,4:6), '--')
            hold off
            grid
            xlabel('Time step')
            xaxis(length(vs.history_low_level));
            legend('v_x', 'v_y', 'v_z', '\omega_x', '\omega_y', '\omega_z')
        end

        function plot_camera(vs)
            %VisualServo.plot_camera Plot camera trajectory
            %
            % VS.plot_camera() plots the camera pose versus time.
            %
            % See also VS.plot_p, VS.plot_vel, VS.plot_error,
            % VS.plot_jcond, VS.plot_z, VS.plot_error, VS.plot_all.

            if isempty(vs.history)
                return
            end
            clf
            % Cartesian camera position vs timestep
            T = SE3( cat(3, vs.history.Tcam) );
            
            subplot(211)
            plot(T.tv');
            xaxis(length(vs.history));
            ylabel('Camera position')
            legend('X', 'Y', 'Z');
            grid
            
            subplot(212)
            plot(T.torpy)
            ylabel('Camera orientation (rad)')
            grid
            xlabel('Time step')
            xaxis(length(vs.history));
            legend('R', 'P', 'Y');
        end

        function plot_jcond(vs)
            %VisualServo.plot_jcond Plot image Jacobian condition
            %
            % VS.plot_jcond() plots image Jacobian condition versus time.
            % Indicates whether the point configuration is close to
            % singular.
            %
            % See also VS.plot_p, VS.plot_vel, VS.plot_error, VS.plot_camera,
            % VS.plot_z, VS.plot_error, VS.plot_all.  
            
            if isempty(vs.history)
                return
            end
            clf

            Jcond = [vs.history.jcond];
            % Image Jacobian condition number vs time
            plot(Jcond);
            grid
            ylabel('Jacobian condition number')
            xlabel('Time step')
            xaxis(length(vs.history));
         end


        function plot_z(vs)
            %VisualServo.plot_z Plot feature depth
            %
            % VS.plot_z() plots feature depth versus time.  If a depth estimator is
            % used it shows true and estimated depth.
            %
            % See also VS.plot_p, VS.plot_vel, VS.plot_error, VS.plot_camera,
            % VS.plot_jcond, VS.plot_error, VS.plot_all.
             if isempty(vs.history)
                return
             end
             
            if strcmp(vs.type, 'point') == 0
                disp('Z-estimator data only computed for point-based IBVS');
                return
            end
            clf
            Zest = [vs.history.Zest];
            Ztrue = [vs.history.Ztrue];
            plot(Ztrue', '-')
            hold on
            set(gca, 'ColorOrderIndex', 1);
            plot(Zest', '--')
            grid
            ylabel('Depth (m)')
            xlabel('Time step')
            xaxis(length(vs.history));
            %legend('true', 'estimate')
        end

        function out = plot_error(vs)
            %VisualServo.plot_error Plot feature error
            %
            % VS.plot_error() plots feature error versus time.
            %
            % See also VS.plot_vel, VS.plot_error, VS.plot_camera,
            % VS.plot_jcond, VS.plot_z, VS.plot_all.
            
            % if isempty(vs.history)
            %     return
            % end
            % clf
            % e = [vs.history.e]';
            % switch vs.type
            %     case 'point'
            %         plot(e(:,1:2:end), 'r');
            %         hold on
            %         plot(e(:,2:2:end), 'b');
            %         hold off
            %         ylabel('Feature error (pixel)')
                    
            %         legend('u', 'v');
            %     otherwise
            %         plot(e);
            %         ylabel('Feature error')
            % end
            % grid
            % xlabel('Time')
            % xaxis(length(vs.history));
            
            % if nargout > 0
            %     out = e;
            % end

            % clearer per-point error plotting20251228修改
            if isempty(vs.history), return; end
            clf
            e = [vs.history.e]';      % Nx(2*npts)
            npts = size(e,2)/2;
            t = 1:size(e,1);
            colors = lines(npts);

            hold on
            legend_entries = {};
            for i = 1:npts
                u = e(:,2*i-1);
                v = e(:,2*i);
                plot(t, u, '-', 'Color', colors(i,:), 'LineWidth', 1.6);
                plot(t, v, '--', 'Color', colors(i,:), 'LineWidth', 1.6);
                legend_entries{end+1} = sprintf('u%d', i);
                legend_entries{end+1} = sprintf('v%d', i);
            end
            hold off
            xlabel('Time'); ylabel('Feature error (pixel)');
            grid on
            xaxis(length(vs.history));
            legend(legend_entries, 'Location','bestoutside', 'Interpreter','none');
            title('Feature error (per point)');
        end

        function plot_all(vs, name, dev)
            %VisualServo.plot_all Plot all trajectory
            %
            % VS.plot_all() plots in separate figures feature values, velocity, 
            % error and camera pose versus time.
            %
            % VS.plot_all(DEV, NAME) writes each plot to a separate file.
            % The name is an SPRINTF format specifier with a %s field that
            % is replaced with a unique per plot suffix.  DEV is the device name
            % passed to the MATLAB print function
            %
            % Example::
            %         vs.plot_all('-depsc', 'eg1%s.eps');
            %
            % See also VS.plot_vel, VS.plot_error, VS.plot_camera,
            % VS.plot_error.
            
            if nargin < 3
                dev = '-depsc';
            end
            if nargin < 2
                name = [];
            end

            figure
            vs.plot_p();
            if ~isempty(name)
                print(gcf, dev, sprintf(name, '-p'));
            end

            figure    % 20251228修改，注释掉原有的plot_vel调用
            vs.plot_vel();
            if ~isempty(name)
                print(gcf, dev, strcat(name, '-vel'));
            end

            figure
            vs.plot_camera();
            if ~isempty(name)
                print(gcf, dev, strcat(name, '-camera'));
            end
            figure
            vs.plot_error();
            if ~isempty(name)
                print(gcf, dev, strcat(name, '-error'));
            end
            
            % optional plots depending on what history was recorded
            if isfield(vs.history, 'Zest')
                figure
                vs.plot_z();
                if ~isempty(name)
                    print(gcf, dev, strcat(name, '-z'));
                end
            end
            
            if isfield(vs.history, 'jcond')
                figure
                vs.plot_jcond();
                if ~isempty(name)
                    print(gcf, dev, strcat(name, '-jcond'));
                end
            end
          
        end

        function display(vs)
            %VisualServo.display Display parameters
            %
            % VS.display() displays the VisualServo parameters in compact single line format.
            %
            % Notes::
            % - This method is invoked implicitly at the command line when the result
            %   of an expression is a VisualServo object and the command has no trailing
            %   semicolon.
            %
            % See also VisualServo.char.

            loose = strcmp( get(0, 'FormatSpacing'), 'loose');
            if loose
                disp(' ');
            end
            disp([inputname(1), ' = '])
            disp( char(vs) );
        end % display()

        function s = char(vs)
            %VisualServo.char Convert to string
            %
            % s = VS.char() is a string showing VisualServo parameters in a compact single line format.
            %
            % See also VisualServo.display.

            s = sprintf('Visual servo object: camera=%s\n  %d iterations, %d history', ...
                vs.camera.name, vs.niter, length(vs.history));
            s = strvcat(s, [['  P= '; '     '; '     '] num2str(vs.P)]);
            if 0
            s = strvcat(s, sprintf('  T0:'));
            s = strvcat(s, [repmat('      ', 4,1) num2str(vs.T0)]);
            s = strvcat(s, sprintf('  C*_T_G:'));
            s = strvcat(s, [repmat('      ', 4,1) num2str(vs.Tf)]);
            else
                if ~isempty(vs.T0)
                    s = strvcat(s, ['  C_T0:   ' trprint(vs.T0, 'fmt', ' %g', 'angvec')]);
                end
                if ~isempty(vs.Tf)
                    s = strvcat(s, ['  C*_T_G: ' trprint(vs.Tf, 'fmt', ' %g', 'angvec')]);
                end
            end
        end
    end
end
