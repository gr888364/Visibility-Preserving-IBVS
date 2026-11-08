#include <iostream>  
#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <mutex>
#include <atomic>
#include <queue>
#include <algorithm>
#include <limits>

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
vpCameraParameters cam(601.82, 602.16, 325.882, 250.568);

// 滤波全局变量
vpColVector v_cam_prev(6, 0.0);
// 世界坐标系4个点（必须固定顺序）
vector<vpPoint> point(4);

// 手眼矩阵
vpHomogeneousMatrix eMc;

// 伺服任务 + 绘图
vpServo task;
vpPlot *plotter = nullptr;
int iter_plot = 0;
// 角点轨迹
std::vector<std::vector<vpImagePoint>> g_trajectories(4);
const int MAX_TRAJECTORY_LEN = 10000;
// 特征
vector<vpFeaturePoint> p(4), pd(4);

// ================= ND：参考 MATLAB IBVS.m 的六维 shunting 增量整形 =================
// MATLAB 逻辑：raw_v_delta = v * dt;
// I_plus  = max(0, K * raw_v_delta); I_minus = max(0, -K * raw_v_delta);
// dStep = -A * Step + (B - Step).*I_plus - (D + Step).*I_minus;
// Step_next = clamp(Step + dStep, -D, B);
// 与 MATLAB 原式的区别仅在最后一步：把 ND 结果投影回原始六维速度方向，
// 防止逐分量的 ND 改变 OG-VFVR 原本的控制方向；且不放大原始速度。
const double ND_K_GAIN = 8.0;
const double ND_A_STEP = 0.4;
const double ND_B_STEP = 0.05;
const double ND_D_STEP = 0.05;

struct NdStepState {
    vpColVector step;
    NdStepState() : step(6, 0.0) {}
    void reset() { step = 0.0; }
};
NdStepState g_nd_step;

static vpColVector nd_smooth_increment_same_direction(const vpColVector &raw_v_cam, double dt)
{
    // 只使用有效视觉帧的正时间步长；无有效输入时保持原指令。
    if (raw_v_cam.size() != 6 || !std::isfinite(dt) || dt <= 1e-9)
        return raw_v_cam;

    const vpColVector raw_delta = raw_v_cam * dt;
    const double delta_norm_sq = raw_delta.sumSquare();
    if (!std::isfinite(delta_norm_sq) || delta_norm_sq <= 1e-24) {
        g_nd_step.reset();
        return raw_v_cam;
    }

    // 完全对应 MATLAB IBVS.m 中的六维 I_plus/I_minus + shunting 状态更新。
    vpColVector matlab_nd_delta(6, 0.0);
    for (unsigned int i = 0; i < 6; ++i) {
        const double u = ND_K_GAIN * raw_delta[i];
        const double I_plus = std::max(0.0, u);
        const double I_minus = std::max(0.0, -u);
        const double curr = g_nd_step.step[i];
        const double d_step = -ND_A_STEP * curr
                            + (ND_B_STEP - curr) * I_plus
                            - (ND_D_STEP + curr) * I_minus;
        const double next = std::max(-ND_D_STEP,
                             std::min(ND_B_STEP, curr + d_step));
        g_nd_step.step[i] = next;
        matlab_nd_delta[i] = next;
    }

    // MATLAB 原版直接将 matlab_nd_delta 用作执行增量；那会改变六维方向。
    // 根据用户要求，只取其沿原始增量方向的非负投影，保持 OG-VFVR 方向。
    double projection = 0.0;
    for (unsigned int i = 0; i < 6; ++i)
        projection += matlab_nd_delta[i] * raw_delta[i];
    const double scale = std::max(0.0,
                         std::min(1.0, projection / delta_norm_sq));
    return raw_v_cam * scale;
}

// ================= ND 前后相机指令位移增量抖动 =================
struct JitterMetrics {
    bool has_previous = false;
    size_t samples = 0;
    vpColVector previous_raw;
    vpColVector previous_nd;
    std::vector<double> raw_jump_norms;
    std::vector<double> nd_jump_norms;

    void break_sequence() { has_previous = false; }
    void reset() {
        has_previous = false;
        samples = 0;
        raw_jump_norms.clear();
        nd_jump_norms.clear();
    }
    void record(const vpColVector &raw_delta, const vpColVector &nd_delta) {
        if (raw_delta.size() != 6 || nd_delta.size() != 6) return;
        if (has_previous) {
            raw_jump_norms.push_back((raw_delta - previous_raw).euclideanNorm());
            nd_jump_norms.push_back((nd_delta - previous_nd).euclideanNorm());
        }
        previous_raw = raw_delta;
        previous_nd = nd_delta;
        has_previous = true;
        ++samples;
    }
    static double sample_std(const std::vector<double> &v) {
        if (v.size() < 2) return std::numeric_limits<double>::quiet_NaN();
        double mean = 0.0;
        for (double x : v) mean += x;
        mean /= static_cast<double>(v.size());
        double variance = 0.0;
        for (double x : v) variance += (x - mean) * (x - mean);
        return std::sqrt(variance / static_cast<double>(v.size() - 1));
    }
    void print() const {
        const double base = sample_std(raw_jump_norms);
        const double nd = sample_std(nd_jump_norms);
        const double reduction = (std::isfinite(base) && base > 1e-12)
            ? 100.0 * (base - nd) / base
            : std::numeric_limits<double>::quiet_NaN();
        std::cout << std::fixed << std::setprecision(8)
                  << "\n========== Increment jitter ==========\n"
                  << "N_samples      = " << samples << "\n"
                  << "sigma_baseline = " << base << "\n"
                  << "sigma_ND       = " << nd << "\n"
                  << "Reduction (%)  = " << reduction << "\n"
                  << "Source         = 6D camera command increments, not measured pose\n"
                  << "================================================\n";
    }
};

// ================= 全局指令（无队列） =================
atomic<bool> start_servo{false};
// 仅用于终端统计与提示，不参与控制
atomic<bool> g_servo_converged{false};
atomic<bool> g_stop_message_printed{false};
atomic<bool> robot_thread_running{true};
JointValue g_current_cmd = {0};
bool g_has_new_cmd = false;
mutex cmd_mutex;

// 虚拟FOV全局变量（必须放外面，不能放if内部）
vector<vpImagePoint> g_last_valid_corners(4);
bool g_has_last_valid = false;

// 用于完全识别不到标定板时，根据最后两帧真实角点估计回拉方向
vector<vpImagePoint> g_prev_valid_corners(4);
bool g_has_prev_valid = false;

vector<vpImagePoint> g_legacy_last_corners(4);
bool g_has_legacy_last = false;

// 安全视野边界：红色安全框，特征点一旦进入该边界外侧/边界上，就生成虚拟特征。
// 注意：虚拟特征生成算法严格按 MATLAB legacy_fov_virtual_uv() 的 Cheng/RVC 逻辑。
const int LEGACY_FOV_MARGIN = 100;
const double LEGACY_FOV_ACTIVE_SPEED_SCALE = 1.0;
// MATLAB legacy logic: do not force a minimum virtual shift.


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



    vpVelocityTwistMatrix eVc(eMc);
    vpColVector v_e = eVc * v_cam;

    // ====================== 已移除：雅可比不在这计算 ======================

    vpColVector dq(6, 0.0);

    cout << "dq [rad/s]: ";
    for (int i = 0; i < 6; i++) {
        cout << fixed << setprecision(6) << dq[i] << "  ";
    }
    cout << endl;

    const double MAX_STEP = 0.003;
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
    const double DT_ROBOT = 0.008;
    static double last_t = vpTime::measureTimeMs();

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

            if (err > convergence_threshold)
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
                    if (!g_stop_message_printed.exchange(true)) {
                        cout << "[JAKA] 相机位姿已收敛 → 平滑停止" << endl;
                    }
                }
            }
        }
        
        // 仅运行且尚未首次收敛时，每250个机器人周期打印一次
        static unsigned int cycle_print_count = 0;
        if (start_servo.load() && !g_servo_converged.load()) {
            if (++cycle_print_count >= 250) {
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

void draw_target_box(vpImage<unsigned char>& I) {
    for (int i=0; i<4; i++) {
        vpImagePoint ip;
        vpMeterPixelConversion::convertPoint(cam, pd[i].get_x(), pd[i].get_y(), ip);
        vpDisplay::displayCross(I, ip, 12, vpColor::red);
        int tx = (int)ip.get_u() + 8;
        int ty = (int)ip.get_v() - 8;
        vpDisplay::displayText(I, ty, tx, to_string(i + 1), vpColor::red);
    }
}

void draw_current_box(vpImage<unsigned char>& I, const vector<vpImagePoint>& corners) {
    if (corners.size()!=4) return;
    for (int i=0; i<4; i++) {
        int j=(i+1)%4;
        vpDisplay::displayLine(I, corners[i], corners[j], vpColor::cyan, 2);
        vpDisplay::displayText(I,
            static_cast<int>(corners[i].get_v()) - 8,
            static_cast<int>(corners[i].get_u()) + 8,
            to_string(i + 1), vpColor::white);
    }
    for (int i = 0; i < 4; ++i) {g_trajectories[i].push_back(corners[i]);
        if (g_trajectories[i].size() > MAX_TRAJECTORY_LEN)
            g_trajectories[i].erase(g_trajectories[i].begin());
        for (size_t k = 1; k < g_trajectories[i].size(); ++k) {
            vpDisplay::displayLine(I,g_trajectories[i][k-1],g_trajectories[i][k],vpColor::green, 1);
        }
    }
}

static vpImagePoint make_image_point_uv(double u, double v)
{
    vpImagePoint ip;
    ip.set_u(u);
    ip.set_v(v);
    return ip;
}

static double image_point_distance(const vpImagePoint& a, const vpImagePoint& b)
{
    const double du = a.get_u() - b.get_u();
    const double dv = a.get_v() - b.get_v();
    return sqrt(du * du + dv * dv);
}

static double min_feature_boundary(const vector<vpImagePoint>& uv, int W, int H)
{
    double d_min = 1e9;
    for (const auto& ip : uv) {
        d_min = min(d_min, ip.get_u());
        d_min = min(d_min, ip.get_v());
        d_min = min(d_min, static_cast<double>(W) - ip.get_u());
        d_min = min(d_min, static_cast<double>(H) - ip.get_v());
    }
    return d_min;
}

static vector<vpImagePoint> predict_image_points(
    const vector<vpImagePoint>& uv,
    const vector<vpImagePoint>& last_uv,
    double steps)
{
    vector<vpImagePoint> predicted = uv;
    for (size_t i = 0; i < uv.size() && i < last_uv.size(); ++i) {
        const double du = uv[i].get_u() - last_uv[i].get_u();
        const double dv = uv[i].get_v() - last_uv[i].get_v();
        predicted[i].set_u(uv[i].get_u() + steps * du);
        predicted[i].set_v(uv[i].get_v() + steps * dv);
    }
    return predicted;
}

// 完全识别不到标定板时使用。
// 根据丢失前最后两帧真实角点估计出界方向，然后反方向构造虚拟特征，
// 让控制器基于虚拟特征误差把标定板拉回视野。
static vector<vpImagePoint> make_lost_tag_pullback_virtual_points(
    const vector<vpImagePoint>& last_uv,
    const vector<vpImagePoint>& prev_uv,
    int W,
    int H,
    int margin)
{
    vector<vpImagePoint> virtual_uv = last_uv;

    if (last_uv.empty() || last_uv.size() != prev_uv.size()) {
        return virtual_uv;
    }

    const double pull_gain = 4.0;

    for (size_t i = 0; i < last_uv.size(); ++i) {
        const double du = last_uv[i].get_u() - prev_uv[i].get_u();
        const double dv = last_uv[i].get_v() - prev_uv[i].get_v();

        double u = last_uv[i].get_u() - pull_gain * du;
        double v = last_uv[i].get_v() - pull_gain * dv;

        // 放入安全框内，给控制器明确的回拉误差方向
        u = std::max((double)margin, std::min((double)(W - margin), u));
        v = std::max((double)margin, std::min((double)(H - margin), v));

        virtual_uv[i].set_u(u);
        virtual_uv[i].set_v(v);
    }

    return virtual_uv;
}

static void translate_points_inside_margin(vector<vpImagePoint>& uv, int W, int H, int margin)
{
    if (uv.empty()) {
        return;
    }

    double min_u = 1e9, min_v = 1e9, max_u = -1e9, max_v = -1e9;
    for (const auto& ip : uv) {
        min_u = min(min_u, ip.get_u());
        min_v = min(min_v, ip.get_v());
        max_u = max(max_u, ip.get_u());
        max_v = max(max_v, ip.get_v());
    }

    double du = 0.0;
    double dv = 0.0;
    if (min_u < margin) {
        du = margin - min_u;
    } else if (max_u > W - margin) {
        du = W - margin - max_u;
    }
    if (min_v < margin) {
        dv = margin - min_v;
    } else if (max_v > H - margin) {
        dv = H - margin - max_v;
    }

    for (auto& ip : uv) {
        ip.set_u(ip.get_u() + du);
        ip.set_v(ip.get_v() + dv);
    }
}

static bool cheng_rvc_virtual_fov(
    const vector<vpImagePoint>& uv,
    const vector<vpImagePoint>& last_uv,
    const vector<vpImagePoint>& uv_star,
    vector<vpImagePoint>& uv_out,
    int& active_index,
    double& shift_u,
    double& shift_v,
    int W,
    int H,
    int margin)
{
    // Cheng/RVC legacy virtual-FOV correction, matched to MATLAB logic.
    uv_out = uv;
    active_index = -1;
    shift_u = 0.0;
    shift_v = 0.0;

    if (uv.empty() || uv.size() != last_uv.size() || uv.size() != uv_star.size()) {
        return false;
    }

    int fov_num = 0;

    // MATLAB order: first scan u row, then v row.
    // If multiple points trigger, the final triggering point becomes active_index.
    for (int j = 0; j < static_cast<int>(uv.size()); ++j) {
        if (uv[j].get_u() <= margin || uv[j].get_u() >= W - margin) {
            ++fov_num;
            active_index = j;
        }
    }

    for (int j = 0; j < static_cast<int>(uv.size()); ++j) {
        if (uv[j].get_v() <= margin || uv[j].get_v() >= H - margin) {
            ++fov_num;
            active_index = j;
        }
    }

    if (fov_num == 0 || active_index < 0) {
        return false;
    }

    // MATLAB: skip the duplicate first result after initialization.
    bool duplicate = true;
    for (size_t j = 0; j < uv.size(); ++j) {
        if (image_point_distance(uv[j], last_uv[j]) > 1e-9) {
            duplicate = false;
            break;
        }
    }
    if (duplicate) {
        return false;
    }

    const vpImagePoint point_a = last_uv[active_index];
    const vpImagePoint point_b = uv[active_index];

    const double radio = image_point_distance(point_b, point_a);
    if (radio < 1e-9) {
        return false;
    }

    const vpImagePoint point_c = make_image_point_uv(
        2.0 * point_b.get_u() - point_a.get_u(),
        2.0 * point_b.get_v() - point_a.get_v());

    const vpImagePoint point_s = uv_star[active_index];
    double du = point_s.get_u() - point_b.get_u();
    double dv = point_s.get_v() - point_b.get_v();
    const double target_norm = sqrt(du * du + dv * dv);
    if (target_norm < 1e-9) {
        return false;
    }

    const vpImagePoint point_d = make_image_point_uv(
        point_b.get_u() + radio * du / target_norm,
        point_b.get_v() + radio * dv / target_norm);

    const vpImagePoint point_e = make_image_point_uv(
        0.5 * (point_d.get_u() + point_c.get_u()),
        0.5 * (point_d.get_v() + point_c.get_v()));

    du = point_e.get_u() - point_b.get_u();
    dv = point_e.get_v() - point_b.get_v();
    const double blend_norm = sqrt(du * du + dv * dv);
    if (blend_norm < 1e-9) {
        return false;
    }

    const vpImagePoint point_f = make_image_point_uv(
        point_b.get_u() + radio * du / blend_norm,
        point_b.get_v() + radio * dv / blend_norm);

    shift_u = point_f.get_u() - point_b.get_u();
    shift_v = point_f.get_v() - point_b.get_v();

    for (auto& ip : uv_out) {
        ip.set_u(ip.get_u() + shift_u);
        ip.set_v(ip.get_v() + shift_v);
    }

    bool virtual_change = false;
    for (const auto& ip : uv_out) {
        if (ip.get_u() <= 0 || ip.get_u() >= W ||
            ip.get_v() <= 0 || ip.get_v() >= H) {
            virtual_change = true;
            break;
        }
    }

    const double delta_u = point_d.get_u() - point_b.get_u();
    const double delta_v = point_d.get_v() - point_b.get_v();

    int iter = 0;
    const int max_iter = 50;

    while (virtual_change && iter < max_iter) {
        for (auto& ip : uv_out) {
            ip.set_u(ip.get_u() + delta_u);
            ip.set_v(ip.get_v() + delta_v);
        }
        shift_u += delta_u;
        shift_v += delta_v;

        int out_num = 0;
        for (const auto& ip : uv_out) {
            if (ip.get_u() < 0 || ip.get_u() > W ||
                ip.get_v() < 0 || ip.get_v() > H) {
                ++out_num;
            }
        }

        if (out_num > 0) {
            for (auto& ip : uv_out) {
                ip.set_u(ip.get_u() + delta_u);
                ip.set_v(ip.get_v() + delta_v);
            }
            shift_u += delta_u;
            shift_v += delta_v;
        } else {
            virtual_change = false;
        }

        ++iter;
    }

    return true;
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
    vpDisplayOpenCV disp(I, 100, 100, "OG-VFVR-IBVS");

    plotter = new vpPlot(4, 700, 700, 750, 100, "OG-VFVR: direction-preserving ND increments");
    plotter->setTitle(0, "No-ND translation increments (m)");
    plotter->setTitle(1, "ND translation increments (m)");
    plotter->setTitle(2, "No-ND rotation increments (rad)");
    plotter->setTitle(3, "ND rotation increments (rad)");
    for (int graph = 0; graph < 4; ++graph) plotter->initGraph(graph, 3);
    // 仅统一绘图标尺，不修改 ND、OG-VFVR 或实际发送的速度指令。
    // 前后平移图共用相同 y 范围（m）；前后旋转图共用相同 y 范围（rad）。
    // 范围需覆盖实验峰值，否则 ViSP vpPlot 可能自动扩展坐标轴。
    const double PLOT_TRANS_ABS_M = 0.10;
    const double PLOT_ROT_ABS_RAD = 0.50;
    const double PLOT_X_MAX = 500.0;
    plotter->initRange(0, 0.0, PLOT_X_MAX, -PLOT_TRANS_ABS_M, PLOT_TRANS_ABS_M);
    plotter->initRange(1, 0.0, PLOT_X_MAX, -PLOT_TRANS_ABS_M, PLOT_TRANS_ABS_M);
    plotter->initRange(2, 0.0, PLOT_X_MAX, -PLOT_ROT_ABS_RAD, PLOT_ROT_ABS_RAD);
    plotter->initRange(3, 0.0, PLOT_X_MAX, -PLOT_ROT_ABS_RAD, PLOT_ROT_ABS_RAD);
    plotter->setUnitY(0, "m");
    plotter->setUnitY(1, "m");
    plotter->setUnitY(2, "rad");
    plotter->setUnitY(3, "rad");
    for (int graph = 0; graph < 2; ++graph) {
        plotter->setLegend(graph, 0, "Delta p_x");
        plotter->setLegend(graph, 1, "Delta p_y");
        plotter->setLegend(graph, 2, "Delta p_z");
    }
    for (int graph = 2; graph < 4; ++graph) {
        plotter->setLegend(graph, 0, "Delta theta_x");
        plotter->setLegend(graph, 1, "Delta theta_y");
        plotter->setLegend(graph, 2, "Delta theta_z");
    }

    task.setServo(vpServo::EYEINHAND_CAMERA);
    task.setInteractionMatrixType(vpServo::MEAN);
    task.setLambda(1.2, 0.4, 0.005);

    for (int i=0; i<4; i++) {
        task.addFeature(p[i], pd[i]);
    }

    vpDetectorAprilTag detector(vpDetectorAprilTag::TAG_36h11);
    detector.setAprilTagPoseEstimationMethod(vpDetectorAprilTag::HOMOGRAPHY_VIRTUAL_VS);

    const double DT_ROBOT = 0.008;
    double t_start = vpTime::measureTimeMs();
    chrono::steady_clock::time_point servo_start_time = chrono::steady_clock::now();
    JitterMetrics jitter_metrics;

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
            // if (dt_actual > 0.05 || dt_actual < 0.001) {
            //     dt_actual = 0.033;
            // }

            const int W = CAM_WIDTH;
            const int H = CAM_HEIGHT;
            const int margin = LEGACY_FOV_MARGIN;
            auto corners = detector.getPolygon(0);
            vector<vpImagePoint> control_corners = corners;
            vector<vpImagePoint> desired_pixels(4);
            bool legacy_active = false;
            int legacy_index = -1;
            double legacy_shift_u = 0.0;
            double legacy_shift_v = 0.0;

            vpDisplay::displayRectangle(I, margin, margin, W - 2 * margin, H - 2 * margin, vpColor::red);
            draw_current_box(I, corners);

            vpHomogeneousMatrix cMo = cMo_vec[0];
            vpHomogeneousMatrix cdMo(vpTranslationVector(0, 0, TAG_SIZE * 6),
                vpRotationMatrix({1, 0, 0, 0, -1, 0, 0, 0, -1}));

            for (int i = 0; i < 4; i++) {
                vpColVector cP;
                point[i].changeFrame(cdMo, cP);
                pd[i].set_x(cP[0] / cP[2]);
                pd[i].set_y(cP[1] / cP[2]);
                pd[i].set_Z(cP[2]);
                vpMeterPixelConversion::convertPoint(cam, pd[i].get_x(), pd[i].get_y(), desired_pixels[i]);

                point[i].changeFrame(cMo, cP);
                p[i].set_Z(cP[2]);
            }

            // ========== 安全视野外：生成虚拟特征 ==========
            // 安全视野是红色框：[margin, W-margin] × [margin, H-margin]
            // 只要任意角点进入红框外侧/压到红框边界，就启用 Cheng/RVC MATLAB legacy 虚拟FOV逻辑。
            const double boundary_dist = min_feature_boundary(corners, W, H);
            bool outside_safe_fov = (boundary_dist <= LEGACY_FOV_MARGIN);

            if (outside_safe_fov) {
                if (g_has_legacy_last) {
                    // 使用 MATLAB legacy_fov_virtual_uv() 对应的 Cheng/RVC 方法生成虚拟特征。
                    // 重要：不 clamp 到红框内，不强制最小位移，不做简单平移。
                    // control_corners 是虚拟特征，只用于控制器计算速度；corners 仍是真实检测角点。
                    legacy_active = cheng_rvc_virtual_fov(
                        corners,
                        g_legacy_last_corners,
                        desired_pixels,
                        control_corners,
                        legacy_index,
                        legacy_shift_u,
                        legacy_shift_v,
                        W,
                        H,
                        LEGACY_FOV_MARGIN);
                } else {
                    // 与 MATLAB 一致：没有上一帧 last_uv 时，只初始化历史点。
                    legacy_active = false;
                }
            }

            // MATLAB legacy FOV 参考点更新策略：
            // 关键：不要在 FOV 已触发后每帧都刷新 last_uv。
            // 否则 point_a 和 point_b 几乎相同，radio 很小，虚拟特征偏移也会很小。
            //
            // 这里保持 last_uv 为“进入安全边界前的最后一帧真实角点”，
            // 这样进入边界后：
            //   point_a = 进入边界前的位置
            //   point_b = 当前边界附近的位置
            //   radio   = 两者距离
            // 更接近 MATLAB 图里的虚拟轨迹效果。
        if (!outside_safe_fov || !g_has_legacy_last) {
            g_legacy_last_corners = corners;
            g_has_legacy_last = true;
        } else {
    const double beta = 0.9;  // 越大偏移越大；建议 0.6~0.9

    for (int i = 0; i < 4; ++i) {
        double u = beta * g_legacy_last_corners[i].get_u()
                 + (1.0 - beta) * corners[i].get_u();
        double v = beta * g_legacy_last_corners[i].get_v()
                 + (1.0 - beta) * corners[i].get_v();

        g_legacy_last_corners[i].set_u(u);
        g_legacy_last_corners[i].set_v(v);
    }
}

            // 保存最后两帧真实检测角点，用于完全识别不到标定板时的回拉逻辑。
            if (g_has_last_valid) {
                g_prev_valid_corners = g_last_valid_corners;
                g_has_prev_valid = true;
            }
            g_last_valid_corners = corners;
            g_has_last_valid = true;

            if (legacy_active) {
                for (int i = 0; i < 4; ++i) {
                    vpDisplay::displayCross(I, control_corners[i], 12, vpColor::yellow);
                }
                for (int i = 0; i < 4; ++i) {
                    int j = (i + 1) % 4;
                    vpDisplay::displayLine(I, control_corners[i], control_corners[j], vpColor::yellow, 2);
                }
            } else {
            }

            for (int i = 0; i < 4; i++) {
                vpFeatureBuilder::create(p[i], cam, control_corners[i]);
            }

            draw_target_box(I);
            vpColVector raw_v_cam = task.computeControlLaw();
            if (legacy_active) {
                raw_v_cam *= LEGACY_FOV_ACTIVE_SPEED_SCALE;
            }

            double error = task.getError().sumSquare();
            const bool collecting_this_frame = start_servo.load();
            // 仅在正常检测到标签、START 且尚未首次收敛时，对幅值进行 ND 整形。
            // 六维控制方向保持与 OG-VFVR 原始输出完全一致。
            vpColVector v_cam = raw_v_cam;
            if (collecting_this_frame && !g_servo_converged.load()
                && error > convergence_threshold) {
                v_cam = nd_smooth_increment_same_direction(raw_v_cam, dt_actual);
                const vpColVector raw_delta = raw_v_cam * dt_actual;
                const vpColVector nd_delta = v_cam * dt_actual;
                jitter_metrics.record(raw_delta, nd_delta);

                vpColVector raw_trans(3), nd_trans(3), raw_rot(3), nd_rot(3);
                for (unsigned int i = 0; i < 3; ++i) {
                    raw_trans[i] = raw_delta[i];
                    nd_trans[i] = nd_delta[i];
                    raw_rot[i] = raw_delta[i + 3];
                    nd_rot[i] = nd_delta[i + 3];
                }
                plotter->plot(0, iter_plot, raw_trans);
                plotter->plot(1, iter_plot, nd_trans);
                plotter->plot(2, iter_plot, raw_rot);
                plotter->plot(3, iter_plot, nd_rot);
                ++iter_plot;
            }
            {
                lock_guard<mutex> lock(g_vis_mutex);
                g_latest_vcam = v_cam;
                g_latest_error = error;
            }

            if (vpDisplay::getClick(I, false)) {
                const bool new_start = !start_servo.load();
                if (new_start) {
                    servo_start_time = chrono::steady_clock::now();
                    jitter_metrics.break_sequence();
                    g_nd_step.reset();
                    g_servo_converged = false;
                    g_stop_message_printed = false;
                    cout << "[视觉伺服] START，开始计时" << endl;
                } else {
                    jitter_metrics.break_sequence();
                    cout << "[视觉伺服] PAUSED" << endl;
                }
                start_servo = new_start;
            }

            // 只在真实检测到标签且首次满足原有收敛判据时打印。
            // 控制判据依然是 task.getError().sumSquare() <= convergence_threshold。
            if (collecting_this_frame && error <= convergence_threshold &&
                !g_servo_converged.exchange(true)) {
                const double elapsed_s = chrono::duration<double>(
                    chrono::steady_clock::now() - servo_start_time).count();
                // 与第二份代码保持一致：相对最终目标像素点计算 RMS 误差。
                double pixel_error_sq_sum = 0.0;
                for (size_t i = 0; i < corners.size() && i < desired_pixels.size(); ++i) {
                    const double du = corners[i].get_u() - desired_pixels[i].get_u();
                    const double dv = corners[i].get_v() - desired_pixels[i].get_v();
                    pixel_error_sq_sum += du * du + dv * dv;
                }
                const double pixel_error_rms = std::sqrt(
                    pixel_error_sq_sum / static_cast<double>(corners.size()));
                std::ostringstream log;
                log << fixed << setprecision(3)
                    << "[视觉伺服] 收敛时间: " << elapsed_s << " s"
                    << "，最终目标像素误差: " << pixel_error_rms << " px";
                cout << log.str() << endl;
                jitter_metrics.print();
            }

            if (start_servo) {
                if (error > convergence_threshold) {
                    vpDisplay::displayText(I, 20, 20, "Status: MOVING", vpColor::cyan);
                } else {
                    vpDisplay::displayText(I, 20, 20, "Status: CONVERGED", vpColor::green);
                }
            } else {
                vpDisplay::displayText(I, 20, 20, "Status: PAUSED", vpColor::orange);
            }

        }
        else
        {
            g_tag_detected = false;
            jitter_metrics.break_sequence(); // 丢失标签的回拉不计入ND抖动统计
            g_nd_step.reset(); // 再次识别时从零状态整形
            int W = CAM_WIDTH;
            int H = CAM_HEIGHT;
            int margin = LEGACY_FOV_MARGIN;
            vpDisplay::displayRectangle(I, margin, margin, W - 2 * margin, H - 2 * margin, vpColor::red);

            // ========== 完全检测不到标签：使用最后两帧真实角点生成回拉虚拟特征 ==========
            if (g_has_last_valid && start_servo) {
                vector<vpImagePoint> virtual_corners = g_last_valid_corners;

                if (g_has_prev_valid) {
                    virtual_corners = make_lost_tag_pullback_virtual_points(
                        g_last_valid_corners,
                        g_prev_valid_corners,
                        W,
                        H,
                        margin);
                } else {
                    // 如果只有最后一帧，没有运动方向，则先把最后角点限制到安全框内
                    for (int i = 0; i < 4; ++i) {
                        double u = virtual_corners[i].get_u();
                        double v = virtual_corners[i].get_v();
                        u = max((double)margin, min((double)(W - margin), u));
                        v = max((double)margin, min((double)(H - margin), v));
                        virtual_corners[i].set_u(u);
                        virtual_corners[i].set_v(v);
                    }
                }

                // 更新期望特征
                vpHomogeneousMatrix cdMo(vpTranslationVector(0, 0, TAG_SIZE * 6),
                    vpRotationMatrix({1, 0, 0, 0, -1, 0, 0, 0, -1}));
                for (int i = 0; i < 4; i++) {
                    vpColVector cP;
                    point[i].changeFrame(cdMo, cP);
                    pd[i].set_x(cP[0] / cP[2]);
                    pd[i].set_y(cP[1] / cP[2]);
                    pd[i].set_Z(cP[2]);
                    p[i].set_Z(TAG_SIZE * 6);
                }

                for (int i = 0; i < 4; i++) {
                    vpFeatureBuilder::create(p[i], cam, virtual_corners[i]);
                }

                vpColVector v_cam = task.computeControlLaw();
                v_cam *= 0.45; // 完全丢失标签时使用虚拟特征回拉，速度仍保持保守

                double error = task.getError().sumSquare();

                {
                    lock_guard<mutex> lock(g_vis_mutex);
                    g_latest_vcam = v_cam;
                    g_latest_error = error;
                }

                for (int i = 0; i < 4; ++i) {
                    vpDisplay::displayCross(I, virtual_corners[i], 12, vpColor::orange);
                }

                draw_target_box(I);
                vpDisplay::displayText(I, 20, 20, "Status: LOST TAG - PULL BACK TO FOV", vpColor::orange);

            } else {
                {
                    lock_guard<mutex> lock(g_vis_mutex);
                    g_latest_vcam = 0;
                    g_latest_error = 1.0;
                }
                vpDisplay::displayText(I, 20, 20, "NO TAG | STOP", vpColor::red);
            }
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