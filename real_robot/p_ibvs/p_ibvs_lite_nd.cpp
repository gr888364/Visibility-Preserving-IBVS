#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <mutex>
#include <atomic>
#include <sstream>
#include <queue>
#include <algorithm>

// VISP 3.5.0 全配套头文件
#include <visp3/visual_features/vpFeaturePoint.h>
#include <visp3/visual_features/vpFeatureBuilder.h>
#include <visp3/vs/vpServo.h>
#include <visp3/detection/vpDetectorAprilTag.h>
#include <visp3/core/vpCameraParameters.h>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpRotationMatrix.h>
#include <visp3/core/vpVelocityTwistMatrix.h>
#include <visp3/gui/vpDisplayOpenCV.h>
#include <visp3/core/vpImageConvert.h>
#include <visp3/core/vpTime.h>
#include <visp3/core/vpMatrix.h>
#include <visp3/gui/vpPlot.h>
#include <visp3/core/vpRzyxVector.h>
#include <visp3/core/vpPixelMeterConversion.h>
#include <unistd.h>

// RealSense
#include <librealsense2/rs.hpp>

// OpenCV
#include <opencv2/opencv.hpp>

// JAKA SDK
extern "C" {
    #include "JAKAZuRobot.h"
    #include "jktypes.h"
}

using namespace std;
using namespace cv;
using namespace rs2;

// ================= 全局宏定义 =================
#define CAM_WIDTH   640
#define CAM_HEIGHT  480

// ================= 全局参数 =================
const char* JAKA_IP = "<ROBOT_IP>";   // set to the IP address of the robot controller
const double TAG_SIZE = 0.0815;
const double convergence_threshold = 0.0000025;
const double pi = 3.1415926;
vpCameraParameters cam(603.89, 604.03, 334.01, 250.33);

// 滤波全局变量
vpColVector v_cam_prev(6, 0.0);
// 世界坐标系4个点（必须固定顺序）
vector<vpPoint> point(4);

// 手眼矩阵
vpHomogeneousMatrix eMc;

// 伺服任务 + 绘图
vpServo task;
vpPlot *plotter = nullptr;
double g_visual_time = 0.0;
// 角点轨迹
std::vector<std::vector<vpImagePoint>> g_trajectories(4);
const int MAX_TRAJECTORY_LEN = 1000;
// 特征
vector<vpFeaturePoint> p(4), pd(4);

// ================= PF-NDRG: potential-field visual reference governor =================
const bool PF_NDRG_ENABLED = true;
const double PF_ATTRACT_GAIN = 2.0;       // IBVS.m: pf_attract_gain
const double PF_REPULSE_GAIN = 10000.0;   // IBVS.m: pf_repulse_gain
const double PF_SAFETY_MARGIN = 50.0;     // IBVS.m: pf_safety_margin, pixels
const double PF_LAMBDA_MIN = 0.08;        // C_IBVS_PF_NDRG.m Proposed PF-NDRG
const double PF_LAMBDA_MAX = 2.55;        // C_IBVS_PF_NDRG.m Proposed PF-NDRG
const double PF_LAMBDA_EXP_RATE = 10.0;   // C_IBVS_PF_NDRG.m Proposed PF-NDRG
const double PF_LAMBDA_EXP_CENTER = 0.74; // C_IBVS_PF_NDRG.m Proposed PF-NDRG
const int PF_LAMBDA_RAMP_STEPS = 300;     // C_IBVS_PF_NDRG.m Proposed PF-NDRG
const bool ND_ENABLED = true;
const double ND_K_GAIN = 8.0;             // IBVS.m: K_gain
const double ND_A_STEP = 0.4;             // IBVS.m: A_step
const double ND_B_STEP = 0.05;            // IBVS.m: B_step
const double ND_D_STEP = 0.05;            // IBVS.m: D_step

struct PfNdrgState {
    bool initialized = false;
    vector<vpImagePoint> s_ref;
    int lambda_step = 0;
    double lambda = PF_LAMBDA_MIN;
};

PfNdrgState g_pf_ndrg;

struct NdStepState {
    bool initialized = false;
    vpColVector step_state;
};

NdStepState g_nd_step;

static void reset_pf_ndrg_state()
{
    g_pf_ndrg.initialized = false;
    g_pf_ndrg.s_ref.clear();
    g_pf_ndrg.lambda_step = 0;
    g_pf_ndrg.lambda = PF_LAMBDA_MIN;
}

static void reset_nd_state()
{
    g_nd_step.initialized = false;
    g_nd_step.step_state.resize(0);
}

// ================= 全局指令（无队列） =================
atomic<bool> start_servo{false};
// 仅用于终端打印的首次收敛标志
atomic<bool> g_servo_converged{false};
atomic<bool> robot_thread_running{true};
JointValue g_current_cmd = {0};
bool g_has_new_cmd = false;
mutex cmd_mutex;

// 新增：视觉线程只传递速度和误差
vpColVector g_latest_vcam(6, 0.0);
double g_latest_error = 1.0;
mutex g_vis_mutex;

// 新增：标签检测标志
atomic<bool> g_tag_detected{false};

//初始化Realsense相机
bool init_realsense(rs2::pipeline& pipeline) {
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, CAM_WIDTH, CAM_HEIGHT, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, CAM_WIDTH, CAM_HEIGHT, RS2_FORMAT_Z16, 30);

    try {
        pipeline.start(cfg);
        cout << "[Realsense] 初始化成功" << endl;
        return true;
    } catch (const rs2::error& e) {
        cerr << "[Realsense] 失败"<< endl;
        return false;
    }
}

//初始化JAKA机械臂
bool init_jaka_robot(JAKAZuRobot& robot, const char* ip) {
    errno_t ret = robot.login_in(ip);
    if (ret != 0) { cerr << "[JAKA] 登录失败" << endl; return false; }
    cout << "[JAKA] 登录成功" << endl;
    this_thread::sleep_for(chrono::seconds(1));

    ret = robot.power_on();
    if (ret != ERR_SUCC) { cerr << "[JAKA] 上电失败" << endl; robot.login_out(); return false; }
    cout << "[JAKA] 上电成功" << endl;
    this_thread::sleep_for(chrono::seconds(2));

    ret = robot.enable_robot();
    if (ret != ERR_SUCC) { cerr << "[JAKA] 使能失败" << endl; robot.power_off(); robot.login_out(); return false; }
    cout << "[JAKA] 使能成功" << endl;

    CartesianPose init_pose = { {114.234, 482.458, 442.573}, {3.14, 0, 0} };
    robot.linear_move(&init_pose, ABS, TRUE, 200, 10, 1, NULL);
    sleep(1);

    ret = robot.edg_init(TRUE, JAKA_IP, 6052, 0);
    if (ret != ERR_SUCC) {
        cerr << "[JAKA] 开启EDG失败" << endl;
        robot.power_off();
        robot.login_out();
        return false;
    }
    cout << "[JAKA] EDG 初始化成功" << endl;

    ret = robot.servo_move_enable(TRUE, FALSE);
    if (ret != ERR_SUCC) {
        cerr << "[JAKA] 伺服模式失败，错误码: " << ret << endl;
        robot.disable_robot();
        robot.power_off();
        robot.login_out();
        return false;
    }
    return true;
}

// DH参数
struct DHParam_Local {
    double alpha, a, d, theta_offset;
};

const DHParam_Local JAKA_DH_CONST[6] = {
    { 0,           0,             119.58/1000.0,         0},
    { 1.5708,      0,                  0,                0},
    { 0,         430.48/1000.0,        0,                0},
    { 0,         368.75/1000.0,   -114.23/1000.0,        0},
    { 1.5708,      0,             113.71/1000.0,         0},
    {-1.5708,      0,             107.48/1000.0,         0}
};

// 几何雅可比
vpMatrix compute_geometric_jacobian(const JointValue &q_curr, vpHomogeneousMatrix &bMe)
{
    vpHomogeneousMatrix T_base_i;
    T_base_i.eye();

    vector<vpColVector> z_list(6), p_list(6);

    for (int i = 0; i < 6; i++) {
        double alpha = JAKA_DH_CONST[i].alpha;
        double a = JAKA_DH_CONST[i].a;
        double d = JAKA_DH_CONST[i].d;
        double theta = q_curr.jVal[i] + JAKA_DH_CONST[i].theta_offset;

        double ct = cos(theta);
        double st = sin(theta);
        double ca = cos(alpha);
        double sa = sin(alpha);

        vpHomogeneousMatrix Ti;
        Ti[0][0] = ct;        Ti[0][1] = -st;       Ti[0][2] = 0;      Ti[0][3] = a;
        Ti[1][0] = st*ca;     Ti[1][1] = ct*ca;     Ti[1][2] = -sa;    Ti[1][3] = -d*sa;
        Ti[2][0] = st*sa;     Ti[2][1] = ct*sa;     Ti[2][2] = ca;     Ti[2][3] = d*ca;
        Ti[3][0] = 0;         Ti[3][1] = 0;         Ti[3][2] = 0;      Ti[3][3] = 1;

        T_base_i = T_base_i * Ti;
        vpRotationMatrix R;
        T_base_i.extract(R);
        z_list[i] = R.getCol(2);
        p_list[i] = T_base_i.getTranslationVector();
    }

    bMe = T_base_i;
    vpColVector p_ee = T_base_i.getTranslationVector();
    vpMatrix J(6, 6);

    for (int i = 0; i < 6; i++) {
        vpColVector z = z_list[i];
        vpColVector p = p_list[i];
        vpColVector dp = p_ee - p;
        vpColVector v = vpColVector::cross(z, dp);

        for (int r = 0; r < 3; r++) {
            J[r][i]   = v[r];
            J[r+3][i] = z[r];
        }
    }

    vpRotationMatrix bRe;
    bMe.extract(bRe);
    vpMatrix T(6,6);
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            T[i][j] = bRe[i][j];
            T[i+3][j+3] = bRe[i][j];
        }
    }
    vpMatrix J_ee = T * J;

    return J_ee;
}

// ================= 【相机线程内计算】关节增量 =================
JointValue compute_servoj_step(const vpColVector &v_cam, JAKAZuRobot &robot, double dt_8ms) {
    JointValue q_curr;
    robot.get_joint_position(&q_curr);

    static vpColVector v_cam_f(6, 0.0);
    double alpha_v = 1;
    v_cam_f = alpha_v * v_cam + (1.0 - alpha_v) * v_cam_f;

    vpVelocityTwistMatrix eVc(eMc);
    vpColVector v_e = eVc * v_cam_f;

    // ====================== 已移除：雅可比不在这计算 ======================

    vpColVector dq(6, 0.0);

    cout << "dq [rad/s]: ";
    for (int i = 0; i < 6; i++) {
        cout << fixed << setprecision(6) << dq[i] << "  ";
    }
    cout << endl;

    const double MAX_STEP = 0.002;
    double max_q_inc = 0;
    for(int i=0; i<6; i++) max_q_inc = max(max_q_inc, abs(dq[i] * dt_8ms));
    if (max_q_inc > MAX_STEP) {
        double scale = MAX_STEP / max_q_inc;
        for(int i=0; i<6; i++) dq[i] *= scale;
    }

    JointValue q_next;
    for (int i = 0; i < 6; i++) {
        q_next.jVal[i] = dq[i] * dt_8ms;
    }

    return q_next;
}

// ================= 机械臂线程：8ms 固定执行 =================
void robot_control_thread(JAKAZuRobot &robot) {
    const double 
    DT_ROBOT = 0.008;
    static double last_t = vpTime::measureTimeMs();
    bool smooth_stop_printed = false; // 每次启动视觉伺服后只打印一次停止提示

    while (robot_thread_running) {
        auto t1 = chrono::steady_clock::now();
        double now_t = vpTime::measureTimeMs();
        double cycle_ms = now_t - last_t;
        last_t = now_t;

        if (start_servo)
        {
            vpColVector v_cam(6, 0.0);
            double err = 0.0;

            {
                lock_guard<mutex> lock(g_vis_mutex);
                v_cam = g_latest_vcam;
                err = g_latest_error;
            }

            if (g_tag_detected && !g_servo_converged.load() && err > convergence_threshold)
            {
                // ======================
                // 【核心移动】8ms线程内实时计算雅可比 + 逆解
                // ======================
                JointValue q_curr;
                robot.get_joint_position(&q_curr);

                static vpColVector v_cam_f(6, 0.0);
                double alpha_v = 1;
                v_cam_f = alpha_v * v_cam + (1.0 - alpha_v) * v_cam_f;

                vpVelocityTwistMatrix eVc(eMc);
                vpColVector v_e = eVc * v_cam_f;

                // ✅ 雅可比矩阵 现在在 8ms 实时线程计算
                vpHomogeneousMatrix bMe;
                vpMatrix J_ee = compute_geometric_jacobian(q_curr, bMe);
                vpMatrix J_inv = J_ee.pseudoInverse();
                vpColVector dq = J_inv * v_e;

                const double MAX_STEP = 0.003;
                double max_q_inc = 0;
                for(int i=0; i<6; i++) max_q_inc = max(max_q_inc, abs(dq[i] * DT_ROBOT));
                if (max_q_inc > MAX_STEP) {
                    double scale = MAX_STEP / max_q_inc;
                    for(int i=0; i<6; i++) dq[i] *= scale;
                }

                JointValue q_cmd;
                for (int i = 0; i < 6; i++) {
                    q_cmd.jVal[i] = dq[i] * DT_ROBOT;
                }

                robot.edg_servo_j(&q_cmd, CONTINUE);
            }
            else
            {
                static vpColVector v_smooth(6, 0.0);
                double decay = 0.85;
                v_smooth = v_smooth * decay;

                // ✅ 停止时也在8ms线程计算雅可比
                JointValue q_curr;
                robot.get_joint_position(&q_curr);
                vpVelocityTwistMatrix eVc(eMc);
                vpColVector v_e = eVc * v_smooth;
                vpHomogeneousMatrix bMe;
                vpMatrix J_ee = compute_geometric_jacobian(q_curr, bMe);
                vpMatrix J_inv = J_ee.pseudoInverse();
                vpColVector dq = J_inv * v_e;

                JointValue q_cmd;
                for (int i = 0; i < 6; i++) {
                    q_cmd.jVal[i] = dq[i] * DT_ROBOT;
                }

                robot.edg_servo_j(&q_cmd, CONTINUE);

                if (v_smooth.infinityNorm() < 1e-4) {
                    v_smooth = 0;
                    if (!smooth_stop_printed) {
                        cout << "[JAKA] 相机位姿已收敛 → 平滑停止" << endl;
                        smooth_stop_printed = true;
                    }
                }
            }
        }
        
        if (!start_servo.load()) {
            smooth_stop_printed = false; // 重新 START 后允许再次打印一次
        }

        // 仅调整终端打印频率，不改变8ms控制周期
        static unsigned int cycle_print_count = 0;
        if (start_servo.load() && !g_servo_converged.load()) {
            if (++cycle_print_count >= 20) {
                cycle_print_count = 0;
                cout << fixed << setprecision(2)
                     << "[机器人线程周期] " << cycle_ms << " ms" << endl;
            }
        } else {
            cycle_print_count = 0;
        }

        auto t2 = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(t2 - t1).count();
        if (elapsed < DT_ROBOT) {
            this_thread::sleep_for(chrono::duration<double>(DT_ROBOT - elapsed));
        }
    }
}

void draw_target_box(vpImage<unsigned char>& I, const vector<vpImagePoint>& uv_star) {
    for (int i=0; i<4; i++) {
        const vpImagePoint& ip = uv_star[i];
        vpDisplay::displayCross(I, ip, 12, vpColor::red);
        int tx = (int)ip.get_u() + 8;
        int ty = (int)ip.get_v() - 8;
        vpDisplay::displayText(I, ty, tx, to_string(i+1).c_str(), vpColor::red);
    }
}

static double clamp_double(double value, double low, double high)
{
    return std::max(low, std::min(value, high));
}

static double min_feature_boundary(const vector<vpImagePoint>& uv, int W, int H)
{
    double d_min = 1e9;
    for (const auto& ip : uv) {
        d_min = std::min(d_min, ip.get_u());
        d_min = std::min(d_min, ip.get_v());
        d_min = std::min(d_min, static_cast<double>(W) - ip.get_u());
        d_min = std::min(d_min, static_cast<double>(H) - ip.get_v());
    }
    return d_min;
}

static double update_pf_adaptive_lambda(bool advance)
{
    const int k_lambda = g_pf_ndrg.lambda_step + 1;
    const int ramp_steps = std::max(1, PF_LAMBDA_RAMP_STEPS);
    const double xi = clamp_double(static_cast<double>(k_lambda - 1) / ramp_steps, 0.0, 1.0);
    const double beta = std::max(PF_LAMBDA_EXP_RATE, 1e-6);
    const double center = clamp_double(PF_LAMBDA_EXP_CENTER, 0.01, 0.99);
    const double sig = 1.0 / (1.0 + std::exp(-beta * (xi - center)));
    const double sig0 = 1.0 / (1.0 + std::exp(beta * center));
    const double sig1 = 1.0 / (1.0 + std::exp(-beta * (1.0 - center)));
    double ramp = (sig - sig0) / std::max(sig1 - sig0, 1e-12);
    ramp = clamp_double(ramp, 0.0, 1.0);
    g_pf_ndrg.lambda = PF_LAMBDA_MIN + (PF_LAMBDA_MAX - PF_LAMBDA_MIN) * ramp;
    if (advance) {
        ++g_pf_ndrg.lambda_step;
    }
    return g_pf_ndrg.lambda;
}

static vpColVector apply_nd_step_smoother(const vpColVector& raw_v_cam, double dt_actual)
{
    if (!ND_ENABLED || dt_actual <= 1e-9) {
        return raw_v_cam;
    }

    const unsigned int n = raw_v_cam.size();
    if (!g_nd_step.initialized || g_nd_step.step_state.size() != n) {
        g_nd_step.step_state.resize(n);
        g_nd_step.step_state = 0;
        g_nd_step.initialized = true;
    }

    vpColVector raw_v_delta = raw_v_cam * dt_actual;
    vpColVector step_next(n);

    for (unsigned int i = 0; i < n; ++i) {
        const double step_curr = g_nd_step.step_state[i];
        const double scaled_input = raw_v_delta[i] * ND_K_GAIN;
        const double i_plus = std::max(0.0, scaled_input);
        const double i_minus = std::max(0.0, -scaled_input);
        const double d_step =
            -ND_A_STEP * step_curr +
            (ND_B_STEP - step_curr) * i_plus -
            (ND_D_STEP + step_curr) * i_minus;
        step_next[i] = clamp_double(step_curr + d_step, -ND_D_STEP, ND_B_STEP);
    }

    g_nd_step.step_state = step_next;
    return step_next / dt_actual;
}

static vector<vpImagePoint> update_pf_ndrg_reference(
    const vector<vpImagePoint>& uv,
    const vector<vpImagePoint>& uv_star,
    double dt,
    int W,
    int H)
{
    if (!g_pf_ndrg.initialized || g_pf_ndrg.s_ref.size() != uv.size()) {
        g_pf_ndrg.s_ref = uv;
        g_pf_ndrg.initialized = true;
    }

    for (size_t i = 0; i < uv.size() && i < uv_star.size(); ++i) {
        const double u_c = uv[i].get_u();
        const double v_c = uv[i].get_v();
        const double u_ref = g_pf_ndrg.s_ref[i].get_u();
        const double v_ref = g_pf_ndrg.s_ref[i].get_v();

        double f_u = PF_ATTRACT_GAIN * (uv_star[i].get_u() - u_ref);
        double f_v = PF_ATTRACT_GAIN * (uv_star[i].get_v() - v_ref);

        if (u_c < PF_SAFETY_MARGIN) {
            const double dist = std::max(u_c, 1e-3);
            const double rep = PF_REPULSE_GAIN *
                std::pow(1.0 / dist - 1.0 / PF_SAFETY_MARGIN, 2.0);
            f_u += rep;
        }
        if (u_c > W - PF_SAFETY_MARGIN) {
            const double dist = std::max(static_cast<double>(W) - u_c, 1e-3);
            const double rep = PF_REPULSE_GAIN *
                std::pow(1.0 / dist - 1.0 / PF_SAFETY_MARGIN, 2.0);
            f_u -= rep;
        }
        if (v_c < PF_SAFETY_MARGIN) {
            const double dist = std::max(v_c, 1e-3);
            const double rep = PF_REPULSE_GAIN *
                std::pow(1.0 / dist - 1.0 / PF_SAFETY_MARGIN, 2.0);
            f_v += rep;
        }
        if (v_c > H - PF_SAFETY_MARGIN) {
            const double dist = std::max(static_cast<double>(H) - v_c, 1e-3);
            const double rep = PF_REPULSE_GAIN *
                std::pow(1.0 / dist - 1.0 / PF_SAFETY_MARGIN, 2.0);
            f_v -= rep;
        }

        const double clamped_u = clamp_double(u_ref, 1.0, static_cast<double>(W));
        const double clamped_v = clamp_double(v_ref, 1.0, static_cast<double>(H));
        const double next_u = clamped_u + f_u * dt;
        const double next_v = clamped_v + f_v * dt;
        g_pf_ndrg.s_ref[i].set_u(next_u);
        g_pf_ndrg.s_ref[i].set_v(next_v);
    }

    return g_pf_ndrg.s_ref;
}

static void apply_virtual_desired_features(
    const vector<vpImagePoint>& s_ref,
    const vector<double>& desired_depths)
{
    for (size_t i = 0; i < s_ref.size() && i < pd.size(); ++i) {
        double x = 0.0;
        double y = 0.0;
        vpPixelMeterConversion::convertPoint(cam, s_ref[i], x, y);
        pd[i].set_x(x);
        pd[i].set_y(y);
        pd[i].set_Z(desired_depths[i]);
    }
}

static void draw_pf_ndrg_reference(
    vpImage<unsigned char>& I,
    const vector<vpImagePoint>& s_ref,
    int W,
    int H)
{
    const int margin = static_cast<int>(PF_SAFETY_MARGIN);
    vpDisplay::displayRectangle(I, margin, margin, W - 2 * margin, H - 2 * margin, vpColor::red);
}

void draw_current_box(vpImage<unsigned char>& I, const vector<vpImagePoint>& corners) {
    if (corners.size()!=4) return;
    for (int i=0; i<4; i++) {
        int j=(i+1)%4;
        vpDisplay::displayLine(I, corners[i], corners[j], vpColor::cyan, 2);
    }
    for (int i = 0; i < 4; ++i) {
        g_trajectories[i].push_back(corners[i]);
        if (g_trajectories[i].size() > MAX_TRAJECTORY_LEN)
            g_trajectories[i].erase(g_trajectories[i].begin());
        for (size_t k = 1; k < g_trajectories[i].size(); ++k) {
            vpDisplay::displayLine(I,g_trajectories[i][k-1],g_trajectories[i][k],vpColor::green, 1);
        }
    }
}

// ================= 视觉线程：一个相机周期发 N 次指令 =================
int main_visual_servo()
{
    point[0].setWorldCoordinates(-TAG_SIZE/2, -TAG_SIZE/2, 0);
    point[1].setWorldCoordinates( TAG_SIZE/2, -TAG_SIZE/2, 0);
    point[2].setWorldCoordinates( TAG_SIZE/2,  TAG_SIZE/2, 0);
    point[3].setWorldCoordinates(-TAG_SIZE/2,  TAG_SIZE/2, 0);

    double raw_eMc[4][4] = {
        { 0.999998,  -0.001822,  0.001204,  -0.034365},
        {-0.001827,  0.999990,   -0.004142, -0.075469},
        {-0.001196,  0.004145,   0.999991,   0.025215},
        {    0,       0,             0,              1}
    };

    for (int i=0;i<4;i++) for(int j=0;j<4;j++) eMc[i][j] = raw_eMc[i][j];

    pipeline pipe;
    if (!init_realsense(pipe)) return -1;

    JAKAZuRobot robot;
    if (!init_jaka_robot(robot, JAKA_IP)) { pipe.stop(); return -1; }

    thread arm_thread(robot_control_thread, std::ref(robot));

    vpImage<unsigned char> I(CAM_HEIGHT, CAM_WIDTH);
    vpDisplayOpenCV disp(I, 100, 100, "P-IBVS + ND");

    plotter = new vpPlot(2, 500, 500, 750, 100, "IBVS with ND - Command Increments");
    plotter->setTitle(0,"ND translation increments (m)");
    plotter->setTitle(1,"ND rotation increments (rad)");
    for (int graph = 0; graph < 2; ++graph) {
        plotter->initGraph(graph, 3);
    }
    plotter->setLegend(0,0,"Delta p_x"); plotter->setLegend(0,1,"Delta p_y"); plotter->setLegend(0,2,"Delta p_z");
    plotter->setLegend(1,0,"Delta theta_x"); plotter->setLegend(1,1,"Delta theta_y"); plotter->setLegend(1,2,"Delta theta_z");

    task.setServo(vpServo::EYEINHAND_CAMERA);
    task.setInteractionMatrixType(vpServo::MEAN);
    task.setLambda(PF_LAMBDA_MIN,PF_LAMBDA_MAX,0.01);

    for (int i=0; i<4; i++) {
        task.addFeature(p[i], pd[i]);
    }

    vpDetectorAprilTag detector(vpDetectorAprilTag::TAG_36h11);
    detector.setAprilTagPoseEstimationMethod(vpDetectorAprilTag::HOMOGRAPHY_VIRTUAL_VS);

    const double DT_ROBOT = 0.008;
    double t_start = vpTime::measureTimeMs();
    chrono::steady_clock::time_point servo_start_time = chrono::steady_clock::now();

    while (true)
    {
        frameset fs = pipe.wait_for_frames();
        frame cf = fs.get_color_frame();
        Mat img(CAM_HEIGHT, CAM_WIDTH, CV_8UC3, (void*)cf.get_data());
        vpImageConvert::convert(img, I);
        vpDisplay::display(I);

        vector<vpHomogeneousMatrix> cMo_vec;
        detector.detect(I, TAG_SIZE, cam, cMo_vec);

        if (detector.getNbObjects() > 0)
        {
            g_tag_detected = true;

            double t_now = vpTime::measureTimeMs();
            double dt_actual = (t_now - t_start) / 1000.0;
            t_start = t_now;
            // 终端不逐帧打印dt_actual，避免干扰周期输出
            if (dt_actual <= 1e-9) {
                dt_actual = 1e-9;
            }
            auto corners = detector.getPolygon(0);
            draw_current_box(I, corners);
            for (int i = 0; i < 4; ++i) {
                vpDisplay::displayText(I, (int)corners[i].get_v()-8, (int)corners[i].get_u()+8, to_string(i+1).c_str(), vpColor::white);
            }

            vpHomogeneousMatrix cMo = cMo_vec[0];
            vpHomogeneousMatrix cdMo(vpTranslationVector(0,0,TAG_SIZE*6), vpRotationMatrix({1,0,0,0,-1,0,0,0,-1}));
            vector<vpImagePoint> uv_star(4);
            vector<vpImagePoint> virtual_uv = corners;
            vector<double> desired_depths(4, TAG_SIZE * 6);
            for (int i=0; i<4; i++) {
                vpColVector cP;
                point[i].changeFrame(cdMo, cP);
                pd[i].set_x(cP[0]/cP[2]); pd[i].set_y(cP[1]/cP[2]); pd[i].set_Z(cP[2]);
                desired_depths[i] = cP[2];
                vpMeterPixelConversion::convertPoint(cam, pd[i].get_x(), pd[i].get_y(), uv_star[i]);
                vpFeatureBuilder::create(p[i], cam, corners[i]);
                point[i].changeFrame(cMo, cP);
                p[i].set_Z(cP[2]);
            }

            bool pf_active = false;
            const bool servo_active = start_servo.load();
            double boundary_dist = min_feature_boundary(corners, CAM_WIDTH, CAM_HEIGHT);
            if (PF_NDRG_ENABLED) {
                if (servo_active) {
                    virtual_uv = update_pf_ndrg_reference(corners, uv_star, dt_actual, CAM_WIDTH, CAM_HEIGHT);
                } else {
                    reset_pf_ndrg_state();
                    reset_nd_state();
                    virtual_uv = corners;
                }
                apply_virtual_desired_features(virtual_uv, desired_depths);
                pf_active = (boundary_dist < PF_SAFETY_MARGIN);
                draw_pf_ndrg_reference(I, virtual_uv, CAM_WIDTH, CAM_HEIGHT);
            }
            const double lambda_now = servo_active ? update_pf_adaptive_lambda(true) : PF_LAMBDA_MIN;
            task.setLambda(lambda_now);
            draw_target_box(I, uv_star);
            vpColVector raw_v_cam = task.computeControlLaw();
            vpColVector v_cam = servo_active ? apply_nd_step_smoother(raw_v_cam, dt_actual) : raw_v_cam;
            double error = task.getError().sumSquare();

            // 仅新增首次收敛的终端统计，不修改原有收敛判据
            if (start_servo.load() &&
                error <= convergence_threshold &&
                !g_servo_converged.exchange(true)) {
                const double converge_s = chrono::duration<double>(
                    chrono::steady_clock::now() - servo_start_time).count();
                // 相对最终目标uv_star计算RMS像素误差，而不是虚拟参考点
                double pixel_error_sq_sum = 0.0;
                for (size_t i = 0; i < corners.size() && i < uv_star.size(); ++i) {
                    const double du = corners[i].get_u() - uv_star[i].get_u();
                    const double dv = corners[i].get_v() - uv_star[i].get_v();
                    pixel_error_sq_sum += du * du + dv * dv;
                }
                const double pixel_error_rms = std::sqrt(
                    pixel_error_sq_sum / static_cast<double>(corners.size()));
                std::ostringstream log;
                log << fixed << setprecision(3)
                    << "[视觉伺服] 收敛时间: " << converge_s << " s"
                    << "，最终目标像素误差: " << pixel_error_rms << " px";
                cout << log.str() << endl;
            }

            // 首次收敛后锁定，向机器人线程发送零速度和零控制误差。
            const bool converged_now = g_servo_converged.load();
            if (converged_now) {
                v_cam = 0;
            }
            {
                lock_guard<mutex> lock(g_vis_mutex);
                g_latest_vcam = start_servo.load() ? v_cam : vpColVector(6, 0.0);
                g_latest_error = converged_now ? 0.0 : error;
            }

            if (vpDisplay::getClick(I, false)) {
                start_servo = !start_servo;
                // 以下只维护终端统计状态，不参与机械臂控制
                if (start_servo.load()) {
                    servo_start_time = chrono::steady_clock::now();
                    g_visual_time = 0.0;

                    // 每次 START 都从零开始采集，并清除上一次的两幅图。
                    if (plotter != nullptr) {
                        plotter->resetPointList(0);
                        plotter->resetPointList(1);
                    }
                    reset_pf_ndrg_state();
                    reset_nd_state();
                }
                // 新一轮 START 清除收敛锁定；清空旧指令，防止重新启动时沿用旧速度。
                if (start_servo.load()) {
                    g_servo_converged.store(false);
                }
                {
                    lock_guard<mutex> lock(g_vis_mutex);
                    g_latest_vcam = 0;
                    g_latest_error = 0.0;
                }
                cout << "[JAKA] 视觉伺服: "
                     << (start_servo.load() ? "START" : "PAUSE") << endl;
            }

            if (start_servo) {
                if (!g_servo_converged.load()) {
                    vpDisplay::displayText(I, 20, 20, "Status: MOVING (EDG)", vpColor::cyan);
                } else {
                    vpDisplay::displayText(I, 20, 20, "Status: CONVERGED", vpColor::green);
                }
                vpDisplay::displayText(I, 40, 20, "Error: " + to_string(error), vpColor::yellow);
            } else {
                vpDisplay::displayText(I, 20, 20, "Status: PAUSED", vpColor::orange);
            }
            vpDisplay::displayText(I, 60, 20,
                string("PF-NDRG: ") + (PF_NDRG_ENABLED ? (pf_active ? "ACTIVE" : "READY") : "OFF") +
                " | lambda=" + to_string(lambda_now) +
                " | boundary=" + to_string(boundary_dist),
                pf_active ? vpColor::red : vpColor::green);

            // 仅在视觉伺服 START 后采集；收敛后两幅图的采样值置零。
            // 图中仍记录 ND 输出速度指令在本视觉周期内的增量，并非实测位移。
            if (start_servo.load()) {
                g_visual_time += dt_actual;

                vpColVector nd_translation(3, 0.0);
                vpColVector nd_rotation(3, 0.0);

                if (!g_servo_converged.load()) {
                    vpColVector nd_delta = v_cam * dt_actual;
                    for (unsigned int i = 0; i < 3; ++i) {
                        nd_translation[i] = nd_delta[i];
                        nd_rotation[i] = nd_delta[i + 3];
                    }
                }

                plotter->plot(0, g_visual_time, nd_translation);
                plotter->plot(1, g_visual_time, nd_rotation);
            }
        }
        else
        {
            g_tag_detected = false;
            reset_pf_ndrg_state();
            reset_nd_state();
            {
                lock_guard<mutex> lock(g_vis_mutex);
                g_latest_vcam = 0;
                g_latest_error = 1.0;
            }

            vpDisplay::displayText(I, 20, 20, "NO TAG", vpColor::red);
        }

        vpDisplay::flush(I);
    }

    robot_thread_running = false;
    if (arm_thread.joinable()) arm_thread.join();

    robot.servo_move_enable(FALSE);
    robot.disable_robot();
    robot.power_off();
    robot.login_out();
    pipe.stop();
    task.kill();
    return 0;
}

int main()
{
    main_visual_servo();
    return 0;
}
