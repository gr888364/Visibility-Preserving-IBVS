// MPC-IBVS 诊断修正版：修复误判收敛与近目标滤波状态，添加控制/求解诊断。
// 注意：未改动 Q/R、预测时域、DH/雅可比、手眼标定；此文件未经真实机械臂编译/上机验证。
// 先在仿真或确保机械臂无法执行运动的条件下查看日志。

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <mutex>
#include <atomic>
#include <queue>
#include <algorithm>
#include <array>
#include <numeric>
#include <memory>
#include <limits>
#include <streambuf>
#include <string>

// OpenCV (先包含，避免与OSQP的EPS宏冲突)
#include <opencv2/opencv.hpp>

// OSQP + Eigen
#include <OsqpEigen/OsqpEigen.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

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

// OpenCV (必须在OSQP之前包含，避免EPS宏冲突)
#include <opencv2/opencv.hpp>

// RealSense
#include <librealsense2/rs.hpp>

// JAKA SDK
extern "C" {
    #include "JAKAZuRobot.h"
    #include "jktypes.h"
}

// 仅过滤 OsqpEigen::Data 的两条重复矩阵提示，保留其他 std::cerr 信息。
// 不改变 OSQP 数据设置、求解器行为或控制指令。
class OsqpWarningFilterBuf : public std::streambuf {
public:
    explicit OsqpWarningFilterBuf(std::streambuf* target) : target_(target) {}
    ~OsqpWarningFilterBuf() override { sync(); }

protected:
    int_type overflow(int_type ch) override {
        if (traits_type::eq_int_type(ch, traits_type::eof()))
            return traits_type::not_eof(ch);
        line_.push_back(traits_type::to_char_type(ch));
        if (traits_type::to_char_type(ch) == '\n') emit_line();
        return ch;
    }
    std::streamsize xsputn(const char* data, std::streamsize count) override {
        for (std::streamsize i = 0; i < count; ++i) overflow(data[i]);
        return count;
    }
    int sync() override {
        if (!line_.empty()) emit_line();
        return target_->pubsync();
    }

private:
    void emit_line() {
        const bool suppressed =
    line_.find("[OsqpEigen::Data::setHessianMatrix]") != std::string::npos ||
    line_.find("[OsqpEigen::Data::setLinearConstraintsMatrix]") != std::string::npos ||
    line_.find("Please use clearHessianMatrix()") != std::string::npos ||
    line_.find("Please use clearLinearConstraintsMatrix()") != std::string::npos;
        // 跳过警告整行及空白行（包括只有空格、\r、\n 的行），避免终端大面积空行。
        const bool blank_line = line_.find_first_not_of(" \t\r\n") == std::string::npos;
        if (!suppressed && !blank_line) {
            target_->sputn(line_.data(), line_.size());
        }
        line_.clear();
    }
    std::streambuf* target_;
    std::string line_;
};

using namespace std;
using namespace cv;
using namespace rs2;

// ================= 全局宏定义 =================
#define CAM_WIDTH   640
#define CAM_HEIGHT  480
#define SAFE_MARGIN 100        // 边缘安全距离100像素

// ================= 全局参数 =================
const char* JAKA_IP = "<ROBOT_IP>";   // set to the IP address of the robot controller
const double TAG_SIZE = 0.0815;
const double CONVERGENCE_RMS_PX = 1.5;// 收敛阈值，像素均方根误差
const double ROTATION_DAMP_RMS_PX = 8.0;
const double pi = 3.1415926;
const double CAM_PX = 601.82;
const double CAM_PY = 602.16;
const double CAM_U0 = 325.882;
const double CAM_V0 = 250.568;
vpCameraParameters cam(CAM_PX, CAM_PY, CAM_U0, CAM_V0);

// 滤波全局变量
vpColVector v_cam_prev(6, 0.0);
// 世界坐标系4个点（必须固定顺序）
vector<vpPoint> point(4);

// 手眼矩阵
vpHomogeneousMatrix eMc;

// 伺服任务 + 绘图
vpServo task;
vpPlot *plotter_mpc = nullptr;
int iter_plot_mpc = 0; // 仅用于两组显示曲线的采样序号
double g_visual_time = 0.0;
// 角点轨迹
std::vector<std::vector<vpImagePoint>> g_trajectories(4);
const int MAX_TRAJECTORY_LEN = 10000;
// 特征
vector<vpFeaturePoint> p(4), pd(4);

// ================= MPC + ND visual servo parameters =================
const bool MPC_ENABLED = true;
const int MPC_HORIZON = 6;                // 减小预测时域，降低累积误差
const double MPC_Q_WEIGHT = 15.0;         // 降低状态误差权重，消除振荡
const double MPC_R_WEIGHT = 6.0;          // 增大控制代价权重，更稳定
const double MPC_SLACK_WEIGHT = 1e5;      // IBVS.m / C_IBVS_PF_NDRG.m
const double MPC_V_MAX = 0.25;             // 降低线速度
const double MPC_W_MAX = 0.25;             // 大幅降低角速度限制，抑制旋转振荡
const double MPC_FOV_MARGIN = 20.0;       // C_IBVS_PF_NDRG.m default MPC margin
const int MPC_SOLVE_REPEATS = 20;         // C_IBVS_PF_NDRG.m MPC-IBVS spec
const bool ND_ENABLED = false;            // G_t (ND shaping) disabled in this variant
const double ND_K_GAIN = 8.0;             // IBVS.m: K_gain
const double ND_A_STEP = 0.4;             // IBVS.m: A_step
const double ND_B_STEP = 0.05;            // IBVS.m: B_step
const double ND_D_STEP = 0.05;            // IBVS.m: D_step

struct NdStepState {
    bool initialized = false;
    vpColVector step_state;
};

NdStepState g_nd_step;

struct MpcState {
    vpColVector tau_prev;
};

MpcState g_mpc_state;

static void reset_nd_state()
{
    g_nd_step.initialized = false;
    g_nd_step.step_state.resize(0);
}

static void reset_mpc_state()
{
    g_mpc_state.tau_prev.resize(6);
    g_mpc_state.tau_prev = 0;
}

// ================= 性能指标（仅统计，不参与控制） =================
struct PerfMetrics {
    std::vector<std::array<double, 6>> velocities;
    std::vector<std::array<double, 8>> pixel_features;

    void reset() {
        velocities.clear();
        pixel_features.clear();
    }

    void add(const vpColVector& v_cam, const vector<vpImagePoint>& corners) {
        if (v_cam.size() != 6 || corners.size() != 4) return;
        std::array<double, 6> v{};
        std::array<double, 8> s{};
        for (int i = 0; i < 6; ++i) v[i] = v_cam[i];
        for (int i = 0; i < 4; ++i) {
            s[2*i] = corners[i].get_u();
            s[2*i+1] = corners[i].get_v();
        }
        velocities.push_back(v);
        pixel_features.push_back(s);
    }

    static double sample_std(const std::vector<double>& values) {
        if (values.size() < 2) return std::numeric_limits<double>::quiet_NaN();
        const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double sum_sq = 0.0;
        for (double x : values) sum_sq += (x-mean)*(x-mean);
        return std::sqrt(sum_sq / (values.size()-1));
    }

    void print() const {
        const size_t n = velocities.size();
        if (n == 0) return;
        const auto& initial = velocities.front();
        const double p_tr = (std::abs(initial[0]) + std::abs(initial[1]) + std::abs(initial[2])) / 3.0;
        const double p_rot = (std::abs(initial[3]) + std::abs(initial[4]) + std::abs(initial[5])) / 3.0;
        std::vector<double> delta_v, delta2_s;
        for (size_t k = 1; k < n; ++k) {
            double d2 = 0.0;
            for (size_t j = 0; j < 6; ++j) {
                const double d = velocities[k][j] - velocities[k-1][j];
                d2 += d*d;
            }
            delta_v.push_back(std::sqrt(d2));
        }
        for (size_t k = 2; k < pixel_features.size(); ++k) {
            double d2 = 0.0;
            for (size_t j = 0; j < 8; ++j) {
                const double d = pixel_features[k][j] - 2.0*pixel_features[k-1][j] + pixel_features[k-2][j];
                d2 += d*d;
            }
            delta2_s.push_back(std::sqrt(d2));
        }
        std::ostringstream out;
        out << std::fixed << std::setprecision(6)
            << "\n========== Performance indices (real robot / MPC-IBVS) ==========\n"
            << "N_conv         = " << n << " (vision updates)\n"
            << "P_II_tr        = " << p_tr << " (m/s)\n"
            << "P_II_rot       = " << p_rot << " (rad/s)\n"
            << "sigma_Delta_v  = " << sample_std(delta_v) << "\n"
            << "sigma_Delta2_s = " << sample_std(delta2_s) << " (pixel coordinates, px)\n"
            << "=====================================================\n";
        std::cout << out.str();
    }
};

// ================= 全局指令（无队列） =================
atomic<bool> start_servo{false};
atomic<bool> robot_thread_running{true};
atomic<bool> g_servo_converged{false};
atomic<bool> g_stop_message_printed{false};
JointValue g_current_cmd = {0};
bool g_has_new_cmd = false;
mutex cmd_mutex;

// 新增：视觉线程只传递速度和误差
vpColVector g_latest_vcam(6, 0.0);
double g_latest_error = std::numeric_limits<double>::infinity();
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

            if (g_tag_detected && err > CONVERGENCE_RMS_PX)
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

                // cout << "[ROBOT] v_cam_f: " << v_cam_f.t() << endl;
                // cout << "[ROBOT] v_e: " << v_e.t() << endl;

                // ✅ 雅可比矩阵 现在在 8ms 实时线程计算
                vpHomogeneousMatrix bMe;
                vpMatrix J_ee = compute_geometric_jacobian(q_curr, bMe);
                vpMatrix J_inv = J_ee.pseudoInverse();
                vpColVector dq = J_inv * v_e;

                // cout << "[ROBOT] dq: " << dq.t() << endl;

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
                    // 标签丢失或尚未获得有效视觉数据都不能判为收敛。
                    if (g_tag_detected.load() &&
                        std::isfinite(err) &&
                        err <= CONVERGENCE_RMS_PX &&
                        g_servo_converged.load() &&
                        !g_stop_message_printed.exchange(true)) {
                        cout << "[JAKA] 视觉误差满足收敛条件 → 停止" << endl;
                    }
                }
            }
        }
        
        static int cycle_print_count = 0;
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
        vpDisplay::displayText(I, ty, tx, to_string(i+1).c_str(), vpColor::red);
    }
}

// 绘制100像素安全红框（危险区域边界）
void draw_safe_red_box(vpImage<unsigned char>& I) {
    vpImagePoint p1(SAFE_MARGIN, SAFE_MARGIN);
    vpImagePoint p2(SAFE_MARGIN, CAM_WIDTH - SAFE_MARGIN);
    vpImagePoint p3(CAM_HEIGHT - SAFE_MARGIN, CAM_WIDTH - SAFE_MARGIN);
    vpImagePoint p4(CAM_HEIGHT - SAFE_MARGIN, SAFE_MARGIN);

    vpDisplay::displayLine(I, p1, p2, vpColor::red, 2);
    vpDisplay::displayLine(I, p2, p3, vpColor::red, 2);
    vpDisplay::displayLine(I, p3, p4, vpColor::red, 2);
    vpDisplay::displayLine(I, p4, p1, vpColor::red, 2);
    vpDisplay::displayText(I, SAFE_MARGIN+10, SAFE_MARGIN+10, "Safe Area", vpColor::red);
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

static vpColVector apply_mpc_velocity_limits(const vpColVector& tau)
{
    if (tau.size() != 6) {
        vpColVector zero_tau(6);
        zero_tau = 0;
        return zero_tau;
    }
    vpColVector limited = tau;
    for (unsigned int i = 0; i < 3; ++i) {
        limited[i] = clamp_double(limited[i], -MPC_V_MAX, MPC_V_MAX);
        limited[i + 3] = clamp_double(limited[i + 3], -MPC_W_MAX, MPC_W_MAX);
    }
    return limited;
}

// 像素坐标向量（与MATLAB一致，s是像素坐标[u1,v1,u2,v2,...]）
static vpColVector image_points_to_vector(const vector<vpImagePoint>& uv)
{
    vpColVector s(uv.size() * 2);
    for (size_t i = 0; i < uv.size(); ++i) {
        s[2 * i] = uv[i].get_u();
        s[2 * i + 1] = uv[i].get_v();
    }
    return s;
}

// 构建像素坐标系下的交互矩阵（与MATLAB B_jacobian 一致）
// MATLAB: B = dt * B_jacobian，其中 B_jacobian 将相机速度映射到像素速度
// 像素速度 = 焦距 * 归一化坐标速度
static vpMatrix build_point_interaction_matrix_pixel(
    const vector<double>& x_norm,  // 归一化坐标 x
    const vector<double>& y_norm,  // 归一化坐标 y
    const vector<double>& depths)
{
    const size_t n = x_norm.size();
    vpMatrix J(n * 2, 6);
    for (size_t i = 0; i < n; ++i) {
        const double x = x_norm[i];
        const double y = y_norm[i];
        const double Z = std::max(depths[i], 1e-6);

        // 像素坐标系下的交互矩阵（归一化交互矩阵乘以焦距）
        // u = px * x + u0 => du = px * dx
        J[2 * i][0] = -CAM_PX / Z;
        J[2 * i][1] = 0.0;
        J[2 * i][2] = CAM_PX * x / Z;
        J[2 * i][3] = CAM_PX * x * y;
        J[2 * i][4] = -CAM_PX * (1.0 + x * x);
        J[2 * i][5] = CAM_PX * y;

        J[2 * i + 1][0] = 0.0;
        J[2 * i + 1][1] = -CAM_PY / Z;
        J[2 * i + 1][2] = CAM_PY * y / Z;
        J[2 * i + 1][3] = CAM_PY * (1.0 + y * y);
        J[2 * i + 1][4] = -CAM_PY * x * y;
        J[2 * i + 1][5] = -CAM_PY * x;
    }
    return J;
}

// 像素坐标转归一化坐标
static void pixel_to_normalized(const vector<vpImagePoint>& uv,
                                 vector<double>& x_norm,
                                 vector<double>& y_norm)
{
    x_norm.resize(uv.size());
    y_norm.resize(uv.size());
    for (size_t i = 0; i < uv.size(); ++i) {
        vpPixelMeterConversion::convertPoint(cam, uv[i], x_norm[i], y_norm[i]);
    }
}

static vpMatrix identity_matrix(unsigned int n)
{
    vpMatrix I(n, n);
    I = 0;
    for (unsigned int i = 0; i < n; ++i) {
        I[i][i] = 1.0;
    }
    return I;
}

// FOV检查使用像素坐标边界（与MATLAB一致）
// FOV 约束以软约束形式写入 QP（见 MPC 求解部分）；此函数对预测特征做像素边界检查
static bool mpc_fov_ok_pixel(const vpColVector& s_pred)
{
    // s_pred 是像素坐标 [u1,v1,u2,v2,...]
    
    for (unsigned int i = 0; i + 1 < s_pred.size(); i += 2) {
        const double u = s_pred[i];
        const double v = s_pred[i + 1];
        if (u < MPC_FOV_MARGIN || u > CAM_WIDTH - MPC_FOV_MARGIN ||
            v < MPC_FOV_MARGIN || v > CAM_HEIGHT - MPC_FOV_MARGIN) {
            return false;
        }
    }
    return true;
    
}

static void add_linear_constraint(
    vector<vpColVector>& A,
    vector<double>& b,
    const vpColVector& row,
    double rhs)
{
    A.push_back(row);
    b.push_back(rhs);
}

static double dot_col_vector(const vpColVector& a, const vpColVector& b)
{
    double value = 0.0;
    const unsigned int n = std::min(a.size(), b.size());
    for (unsigned int i = 0; i < n; ++i) {
        value += a[i] * b[i];
    }
    return value;
}

static vpColVector projected_soft_qp(
    const vpMatrix& H,
    const vpColVector& f,
    const vector<vpColVector>& A,
    const vector<double>& b,
    const vpColVector& z0,
    int iterations)
{
    vpColVector z = z0;
    const double step = 1e-3;  // 增大步长加快收敛
    const double penalty = 500.0;  // 增大惩罚系数

    for (int it = 0; it < iterations; ++it) {
        vpColVector grad = H * z + f;
        for (size_t c = 0; c < A.size(); ++c) {
            const double violation = dot_col_vector(A[c], z) - b[c];
            if (violation > 0.0) {
                grad += A[c] * (penalty * violation);
            }
        }
        z -= grad * step;

        for (size_t c = 0; c < A.size(); ++c) {
            const double violation = dot_col_vector(A[c], z) - b[c];
            if (violation > 0.0) {
                const double norm2 = A[c].sumSquare();
                if (norm2 > 1e-12) {
                    z -= A[c] * (violation / norm2);
                }
            }
        }
    }

    return z;
}

// 解析求解单步无约束MPC（H*u = -f的最小二乘解）
static vpColVector solve_mpc_analytic(
    const vpMatrix& H,
    const vpColVector& f,
    const vpColVector& lb,
    const vpColVector& ub)
{
    // H*u + f = 0  =>  u = -H^{-1}*f (如果H可逆)
    vpMatrix H_inv = H.pseudoInverse();
    vpColVector u = -H_inv * f;

    // 投影到框约束
    for (unsigned int i = 0; i < u.size(); ++i) {
        u[i] = clamp_double(u[i], lb[i], ub[i]);
    }

    return u;
}

// 将像素坐标转换为归一化坐标状态向量（与v4/v6一致）
static vpColVector pixel_to_normalized_state(
    const vector<vpImagePoint>& uv)
{
    vpColVector s(uv.size() * 2);
    for (size_t i = 0; i < uv.size(); ++i) {
        double x, y;
        vpPixelMeterConversion::convertPoint(cam, uv[i], x, y);
        s[2 * i] = x;
        s[2 * i + 1] = y;
    }
    return s;
}

// 构建归一化坐标系下的交互矩阵（与v4/v6一致）
// 状态 s = [x1, y1, x2, y2, x3, y3, x4, y4]^T
// 状态导数 ds = L * v_cam
static vpMatrix build_interaction_matrix_normalized(
    const vector<double>& x_norm,
    const vector<double>& y_norm,
    const vector<double>& depths)
{
    const size_t n = x_norm.size();  // n=4
    vpMatrix L(n * 2, 6);
    for (size_t i = 0; i < n; ++i) {
        const double x = x_norm[i];
        const double y = y_norm[i];
        const double Z = std::max(depths[i], 1e-6);
        const int r = 2 * i;

        // 归一化坐标下的交互矩阵（与v4/v6代码一致）
        L[r][0] = -1.0 / Z;
        L[r][1] =  0.0;
        L[r][2] =  x / Z;
        L[r][3] =  x * y;
        L[r][4] = -(1.0 + x * x);
        L[r][5] =  y;

        L[r+1][0] =  0.0;
        L[r+1][1] = -1.0 / Z;
        L[r+1][2] =  y / Z;
        L[r+1][3] =  1.0 + y * y;
        L[r+1][4] = -x * y;
        L[r+1][5] = -x;
    }
    return L;
}

// 使用ViSP feature计算交互矩阵（与v4完全一致）
static vpMatrix build_interaction_matrix_from_features(
    const vector<vpFeaturePoint>& features)
{
    const size_t n = features.size();  // n=4
    vpMatrix L(n * 2, 6);
    for (size_t i = 0; i < n; ++i) {
        // 创建副本以调用interaction()
        vpFeaturePoint feature_copy = features[i];
        vpMatrix L_i = feature_copy.interaction();
        const int r = 2 * i;
        // 复制2x6的交互矩阵
        for (unsigned int row = 0; row < 2; row++) {
            for (unsigned int col = 0; col < 6; col++) {
                L[r + row][col] = L_i[row][col];
            }
        }
    }
    return L;
}

// MPC求解器 - 严格遵循v4/v6实现
// 使用ViSP feature的interaction()方法，A=I, B=Ts*L
static vpColVector compute_mpc_camera_velocity(
    const vector<vpFeaturePoint>& p_features,  // 当前特征点
    const vector<vpFeaturePoint>& pd_features, // 期望特征点
    double dt_actual,
    bool& mpc_ok)
{
    // ================= MATLAB一致版本：像素状态 + 像素交互矩阵 + 像素FOV软约束 =================
    // MATLAB: s = [u1; v1; u2; v2; ...]
    //         B = dt * B_jacobian，其中B_jacobian把相机速度映射到像素速度
    const int n_points = static_cast<int>(p_features.size());
    const int n_state = n_points * 2;
    const int n_ctrl = 6;
    const int Np = MPC_HORIZON;

    vpColVector s(n_state);
    vpColVector s_target(n_state);
    vector<double> x_norm(n_points), y_norm(n_points), depths(n_points);

    for (int i = 0; i < n_points; i++) {
        const double x = p_features[i].get_x();
        const double y = p_features[i].get_y();
        const double xd = pd_features[i].get_x();
        const double yd = pd_features[i].get_y();

        x_norm[i] = x;
        y_norm[i] = y;
        depths[i] = std::max(p_features[i].get_Z(), 1e-6);

        // 当前/期望状态都转成像素坐标，与MATLAB s、s_target一致
        s[2*i]     = CAM_PX * x  + CAM_U0;
        s[2*i + 1] = CAM_PY * y  + CAM_V0;
        s_target[2*i]     = CAM_PX * xd + CAM_U0;
        s_target[2*i + 1] = CAM_PY * yd + CAM_V0;
    }

    // 像素坐标系下的交互矩阵：像素速度 = L_pixel * v_cam
    const vpMatrix L = build_point_interaction_matrix_pixel(x_norm, y_norm, depths);

    // 离散模型: A = I, B = dt * L_pixel
    vector<vpMatrix> A_powers(Np + 1);
    for (int i = 0; i <= Np; ++i) {
        A_powers[i] = identity_matrix(n_state);
    }

    vpMatrix B = L * dt_actual;

    // 构建预测矩阵 M_s 和 M_tau，与MATLAB mpcqp_generate_cost一致
    vpMatrix M_s(n_state * Np, n_state);
    M_s = 0;
    vpMatrix M_tau(n_state * Np, n_ctrl * Np);
    M_tau = 0;

    for (int i = 1; i <= Np; ++i) {
        for (int row = 0; row < n_state; ++row) {
            for (int col = 0; col < n_state; ++col) {
                M_s[(i-1)*n_state + row][col] = A_powers[i][row][col];
            }
        }

        for (int j = 1; j <= i; ++j) {
            vpMatrix AB = A_powers[i-j] * B;
            for (int row = 0; row < n_state; ++row) {
                for (int col = 0; col < n_ctrl; ++col) {
                    M_tau[(i-1)*n_state + row][(j-1)*n_ctrl + col] = AB[row][col];
                }
            }
        }
    }

    vpColVector S_target(n_state * Np);
    for (int i = 0; i < Np; ++i) {
        for (int j = 0; j < n_state; ++j) {
            S_target[i * n_state + j] = s_target[j];
        }
    }

    // Q/R权重：像素状态下按焦距缩放Q，等价于原来的归一化坐标代价
    // e_u = px * e_x, e_v = py * e_y，因此 Q_u = Q / px^2, Q_v = Q / py^2
    // R保持原C++控制效果：平移用MPC_R_WEIGHT，旋转用MPC_R_WEIGHT * 0.1
    vpMatrix Q_blk(n_state * Np, n_state * Np);
    Q_blk = 0;
    vpMatrix R_blk(n_ctrl * Np, n_ctrl * Np);
    R_blk = 0;

    const double q_u = MPC_Q_WEIGHT / (CAM_PX * CAM_PX);
    const double q_v = MPC_Q_WEIGHT / (CAM_PY * CAM_PY);
    for (int i = 0; i < Np; ++i) {
        for (int pidx = 0; pidx < n_points; ++pidx) {
            Q_blk[i * n_state + 2 * pidx][i * n_state + 2 * pidx] = q_u;
            Q_blk[i * n_state + 2 * pidx + 1][i * n_state + 2 * pidx + 1] = q_v;
        }
        for (int j = 0; j < n_ctrl; ++j) {
            double r_weight = (j < 3) ? MPC_R_WEIGHT : (MPC_R_WEIGHT * 0.1);
            R_blk[i * n_ctrl + j][i * n_ctrl + j] = r_weight;
        }
    }

    vpMatrix H_tau = (M_tau.t() * Q_blk * M_tau + R_blk) * 2.0;
    vpColVector f_tau = M_tau.t() * Q_blk * (M_s * s - S_target) * 2.0;

    H_tau = (H_tau + H_tau.t()) * 0.5;
    for (int i = 0; i < H_tau.getRows(); ++i) {
        H_tau[i][i] += 1e-9;
    }

    // ================= OSQP问题：z = [tau_seq; slack] =================
    const int n_tau = n_ctrl * Np;
    const int slack_idx = n_tau;
    const int n_var = n_tau + 1;

    const int n_vel_con = n_tau;
    const int n_fov_con = 4 * n_points * Np;
    const int n_slack_con = 1;
    const int n_con = n_vel_con + n_fov_con + n_slack_con;

    // Hessian: blkdiag(H_tau, 2*MPC_SLACK_WEIGHT)
    std::vector<Eigen::Triplet<double>> H_triplets;
    for (int i = 0; i < H_tau.getRows(); ++i) {
        for (int j = 0; j < H_tau.getCols(); ++j) {
            if (std::abs(H_tau[i][j]) > 1e-12) {
                H_triplets.emplace_back(i, j, H_tau[i][j]);
            }
        }
    }
    H_triplets.emplace_back(slack_idx, slack_idx, 2.0 * MPC_SLACK_WEIGHT);

    Eigen::SparseMatrix<double> H_eigen(n_var, n_var);
    H_eigen.setFromTriplets(H_triplets.begin(), H_triplets.end());
    H_eigen.makeCompressed();

    Eigen::VectorXd f_eigen(n_var);
    f_eigen.setZero();
    for (int i = 0; i < n_tau; ++i) {
        f_eigen[i] = f_tau[i];
    }

    // OSQP: lb <= A*z <= ub
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<Eigen::Triplet<double>> A_triplets;
    Eigen::VectorXd lb_eigen(n_con), ub_eigen(n_con);
    lb_eigen.setConstant(-inf);
    ub_eigen.setConstant(inf);

    int row_id = 0;

    // 速度约束：-limit <= tau_k <= limit，等价MATLAB A_vel/b_vel
    for (int i = 0; i < Np; ++i) {
        for (int j = 0; j < n_ctrl; ++j) {
            const int col = i * n_ctrl + j;
            const double limit = (j < 3) ? MPC_V_MAX : MPC_W_MAX;
            A_triplets.emplace_back(row_id, col, 1.0);
            lb_eigen[row_id] = -limit;
            ub_eigen[row_id] =  limit;
            ++row_id;
        }
    }

    // FOV软约束，与MATLAB mpcqp_fov_constraints一致：
    // u_pred <= u_max + slack
    // u_pred >= u_min - slack
    // v_pred <= v_max + slack
    // v_pred >= v_min - slack
    double margin = std::max(0.0, MPC_FOV_MARGIN);
    double u_min = margin;
    double u_max = CAM_WIDTH - margin;
    double v_min = margin;
    double v_max = CAM_HEIGHT - margin;

    if (u_min > u_max) {
        u_min = 0.0;
        u_max = CAM_WIDTH;
    }
    if (v_min > v_max) {
        v_min = 0.0;
        v_max = CAM_HEIGHT;
    }

    vpColVector offset = M_s * s;

    auto add_fov_upper_row = [&](int state_row, double rhs_base, double sign) {
        // sign=+1:  state <= upper  ->  M_tau*tau - slack <= upper - offset
        // sign=-1: -state <= -lower -> -M_tau*tau - slack <= -lower + offset
        for (int c = 0; c < n_tau; ++c) {
            const double value = sign * M_tau[state_row][c];
            if (std::abs(value) > 1e-12) {
                A_triplets.emplace_back(row_id, c, value);
            }
        }
        A_triplets.emplace_back(row_id, slack_idx, -1.0);
        ub_eigen[row_id] = rhs_base;
        ++row_id;
    };

    for (int k = 0; k < Np; ++k) {
        const int base = k * n_state;
        for (int pidx = 0; pidx < n_points; ++pidx) {
            const int u_row = base + 2 * pidx;
            const int v_row = base + 2 * pidx + 1;

            add_fov_upper_row(u_row,  u_max - offset[u_row],  1.0);
            add_fov_upper_row(u_row, -u_min + offset[u_row], -1.0);
            add_fov_upper_row(v_row,  v_max - offset[v_row],  1.0);
            add_fov_upper_row(v_row, -v_min + offset[v_row], -1.0);
        }
    }

    // slack >= 0
    A_triplets.emplace_back(row_id, slack_idx, 1.0);
    lb_eigen[row_id] = 0.0;
    ub_eigen[row_id] = inf;
    ++row_id;

    Eigen::SparseMatrix<double> A_eigen(n_con, n_var);
    A_eigen.setFromTriplets(A_triplets.begin(), A_triplets.end());
    A_eigen.makeCompressed();

    // 创建OSQP求解器
    static std::unique_ptr<OsqpEigen::Solver> solver;
    static bool solver_initialized = false;

    if (!solver_initialized) {
        solver = std::make_unique<OsqpEigen::Solver>();
        solver->settings()->setWarmStart(true);
        solver->settings()->setVerbosity(false);
        solver->settings()->setAbsoluteTolerance(1e-6);
        solver->settings()->setRelativeTolerance(1e-6);
        solver->settings()->setMaxIteration(4000);
        solver_initialized = true;
    }

    solver->data()->setNumberOfVariables(n_var);
    solver->data()->setNumberOfConstraints(n_con);
    solver->data()->setHessianMatrix(H_eigen);
    solver->data()->setGradient(f_eigen);
    solver->data()->setLinearConstraintsMatrix(A_eigen);
    solver->data()->setLowerBound(lb_eigen);
    solver->data()->setUpperBound(ub_eigen);

    vpColVector tau(6);
    tau = 0;

    mpc_ok = false;
    if (solver->initSolver()) {
        if (solver->solveProblem() == OsqpEigen::ErrorExitFlag::NoError) {
            Eigen::VectorXd solution = solver->getSolution();
            for (int i = 0; i < 6; ++i) {
                tau[i] = solution[i];
            }
            mpc_ok = true;
        } else {
            cout << "[MPC] OSQP solve failed, using zero control" << endl;
        }
    } else {
        cout << "[MPC] OSQP init failed, using zero control" << endl;
    }

    // // 诊断：判断小速度是优化目标导致，还是求解失败导致。
    // // 这些指标不直接用于运动控制。
    // static unsigned long qp_diag_counter = 0;
    // if (++qp_diag_counter % 30 == 0) {
    //     std::cout << std::scientific << std::setprecision(6)
    //               << "[QP-DIAG] ok=" << mpc_ok
    //               << " dt=" << dt_actual
    //               << " error_norm_px=" << (s - s_target).euclideanNorm()
    //               << " f_norm=" << f_tau.euclideanNorm()
    //               << " H00=" << H_tau[0][0]
    //               << " f0=" << f_tau[0]
    //               << " tau_norm=" << tau.euclideanNorm()
    //               << std::endl;
    // }
    solver->clearSolver();

    g_mpc_state.tau_prev = tau;

    return tau;
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
    // AprilTag检测器角点顺序：0=左上, 1=右上, 2=右下, 3=左下
    // 特征点顺序与v6一致（左下、右下、右上、左上）
    point[0].setWorldCoordinates(-TAG_SIZE/2, -TAG_SIZE/2, 0);  // 左下
    point[1].setWorldCoordinates( TAG_SIZE/2, -TAG_SIZE/2, 0);  // 右下
    point[2].setWorldCoordinates( TAG_SIZE/2,  TAG_SIZE/2, 0);  // 右上
    point[3].setWorldCoordinates(-TAG_SIZE/2,  TAG_SIZE/2, 0);  // 左上

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
    vpDisplayOpenCV disp(I, 100, 100, "MPC-IBVS");

    // 参考第二份代码：仅显示8维视觉特征误差与6维相机速度
    plotter_mpc = new vpPlot(2, 500, 500, 750, 100, "Plot");
    plotter_mpc->setTitle(0, "Visual features error");
    plotter_mpc->setTitle(1, "Camera velocities");
    plotter_mpc->initGraph(0, 8);
    plotter_mpc->initGraph(1, 6);
    plotter_mpc->setLegend(1, 0, "vc_x"); plotter_mpc->setLegend(1, 1, "vc_y"); plotter_mpc->setLegend(1, 2, "vc_z");
    plotter_mpc->setLegend(1, 3, "wc_x"); plotter_mpc->setLegend(1, 4, "wc_y"); plotter_mpc->setLegend(1, 5, "wc_z");

    task.setServo(vpServo::EYEINHAND_CAMERA);
    task.setInteractionMatrixType(vpServo::MEAN);
    task.setLambda(2.0);

    for (int i=0; i<4; i++) {
        task.addFeature(p[i], pd[i]);
    }

    vpDetectorAprilTag detector(vpDetectorAprilTag::TAG_36h11);
    detector.setAprilTagPoseEstimationMethod(vpDetectorAprilTag::HOMOGRAPHY_VIRTUAL_VS);

    const double DT_ROBOT = 0.008;
    double t_start = vpTime::measureTimeMs();
    auto servo_start_time = chrono::steady_clock::now();
    PerfMetrics perf_metrics;

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
            //cout << "dt_actual: " << dt_actual << endl;
            if (dt_actual <= 1e-9) {
                dt_actual = 1e-9;
            }

            auto corners = detector.getPolygon(0);
            draw_current_box(I, corners);
            for (int i = 0; i < 4; ++i) {
                vpDisplay::displayText(I, (int)corners[i].get_v()-8, (int)corners[i].get_u()+8, to_string(i+1).c_str(), vpColor::white);
            }

            vpHomogeneousMatrix cMo = cMo_vec[0];
            // 期望位姿：相机正对标签，Z轴指向标签，距离6倍标签大小
            // 使用RX=180°旋转（与v6一致，匹配相机坐标系）
            // 旋转矩阵: [1,0,0; 0,-1,0; 0,0,-1] 表示绕X轴旋转180度
            vpHomogeneousMatrix cdMo(vpTranslationVector(0, 0, TAG_SIZE*6),
                                     vpRotationMatrix({1,0,0,0,-1,0,0,0,-1}));
            vector<vpImagePoint> uv_star(4);
            vector<double> current_depths(4);  // 使用当前深度计算交互矩阵（与v4一致）
            for (int i=0; i<4; i++) {
                vpColVector cP;
                // 计算期望特征
                point[i].changeFrame(cdMo, cP);
                pd[i].set_x(cP[0]/cP[2]); pd[i].set_y(cP[1]/cP[2]); pd[i].set_Z(cP[2]);
                vpMeterPixelConversion::convertPoint(cam, pd[i].get_x(), pd[i].get_y(), uv_star[i]);
                // 计算当前特征和当前深度
                vpFeatureBuilder::create(p[i], cam, corners[i]);
                point[i].changeFrame(cMo, cP);
                p[i].set_Z(cP[2]);
                current_depths[i] = cP[2];  // 当前实际深度
            }

            const bool servo_active = start_servo.load();
            if (servo_active) {
                g_visual_time += dt_actual;
            }
            double boundary_dist = min_feature_boundary(corners, CAM_WIDTH, CAM_HEIGHT);
            draw_target_box(I);
            draw_safe_red_box(I);
            bool mpc_ok = true;
            vpColVector raw_v_cam = compute_mpc_camera_velocity(p, pd, dt_actual, mpc_ok);
            vpColVector v_cam = raw_v_cam;
            // 单独增加角速度增益


            //v_cam[3];  // 注意：z轴方向相反，需要翻转
            // 打印MPC输出的原始速度，用于诊断绕圈问题
            //cout << "[MPC RAW] v_cam: " << raw_v_cam.t() << endl;

            // 不抑制任何方向的运动，让MPC自由控制全部6个自由度同时收敛
            // 移除分阶段控制和旋转阻尼，确保四个点同时收敛

            // 调试：打印期望和当前位置
            static int dbg_cnt = 0;
            if (start_servo.load() && !g_servo_converged.load() && ++dbg_cnt % 150 == 0) {
                cout << "[DEBUG] target_center: (" << (uv_star[0].get_u()+uv_star[2].get_u())/2
                     << ", " << (uv_star[0].get_v()+uv_star[2].get_v())/2 << ")" << endl;
                cout << "[DEBUG] current_center: (" << (corners[0].get_u()+corners[2].get_u())/2
                     << ", " << (corners[0].get_v()+corners[2].get_v())/2 << ")" << endl;
            }

            //cout << "[MPC OUT] v_cam: " << v_cam.t() << ", mpc_ok: " << mpc_ok << endl;

            // 不对 MPC 输出做后验置零，FOV 由 QP 中的软约束处理
            // if (!mpc_ok && boundary_dist <= MPC_FOV_MARGIN) {
            //     v_cam = 0;
            //     raw_v_cam = 0;
            // }
            // MATLAB中 s 是像素坐标，误差直接是像素误差
            vpColVector feature_error = image_points_to_vector(corners) - image_points_to_vector(uv_star);
            double pixel_error = std::sqrt(feature_error.sumSquare() / std::max(1u, feature_error.size()));
            // 在视觉伺服循环中，计算完v_cam后添加
            // 在全局变量区域添加
// // 保留跨帧滤波状态；只在误差较小时进行滤波。
// // 这里是输出指令滤波，与 plot 中 raw_v_cam 的原始 MPC 输出不同。
// static vpColVector v_cam_filtered(6, 0.0);
// if (pixel_error < 25.0 && mpc_ok) {
//     const double alpha = 0.2;
//     for (int i = 0; i < 6; ++i) {
//         v_cam_filtered[i] = alpha * v_cam[i] + (1.0 - alpha) * v_cam_filtered[i];
//         v_cam[i] = v_cam_filtered[i];
//     }
// } else {
//     v_cam_filtered = mpc_ok ? v_cam : vpColVector(6, 0.0);
// }
// MPC 求解失败时禁止采用上一帧速度。
// if (!mpc_ok) {
//     v_cam = 0;
//     v_cam_filtered = 0;
// }

// // 仅诊断，不影响控制。请首先在禁用机器人运动的情况下观察日志。
// static unsigned long mpc_diag_counter = 0;
// if (start_servo.load() && (++mpc_diag_counter % 30 == 0)) {
//     std::cout << std::scientific << std::setprecision(6)
//               << "[MPC-DIAG] rms_px=" << pixel_error
//               << " dt=" << dt_actual
//               << " solver_ok=" << mpc_ok
//               << " raw_norm=" << raw_v_cam.euclideanNorm()
//               << " cmd_norm=" << v_cam.euclideanNorm()
//               << " raw_v=" << raw_v_cam.t() << std::endl;
// }
            // 性能指标：从 START 后第一帧开始记录；收敛帧也记录。
            // 使用控制器的相机速度（MPC 输出经过本处滤波，但尚未被停止逻辑置零），
            // 像素特征使用真实 AprilTag 角点；不使用绘图的人工零值。
            if (start_servo.load() && !g_servo_converged.load() && mpc_ok) {
                perf_metrics.add(v_cam, corners);
                if (pixel_error <= CONVERGENCE_RMS_PX &&
                    !g_servo_converged.exchange(true)) {
                    const double converge_s = chrono::duration<double>(
                        chrono::steady_clock::now() - servo_start_time).count();
                    std::cout << std::fixed << std::setprecision(3)
                        << "[视觉伺服] 收敛时间: " << converge_s
                        << " s，最终RMS像素误差: " << pixel_error << " px\n";
                    perf_metrics.print();
                }
            }
            // 调试输出
            static int error_print_count = 0;
            if (start_servo.load() && !g_servo_converged.load() &&++error_print_count >= 150) {
                error_print_count = 0;
                cout << "[DEBUG] pixel_error: " << pixel_error
                    << " px, boundary_dist: " << boundary_dist << endl;
            }
            if (pixel_error <= CONVERGENCE_RMS_PX) {
                v_cam = 0;
                reset_nd_state();
            }

            {
                lock_guard<mutex> lock(g_vis_mutex);
                g_latest_vcam = v_cam;
                g_latest_error = pixel_error;  // 像素误差
            }

        if (vpDisplay::getClick(I, false)) {
            start_servo = 1;

            if (start_servo.load()) {
                servo_start_time = chrono::steady_clock::now();
                g_servo_converged.store(false);
                g_stop_message_printed.store(false);

                g_visual_time = 0.0;
                reset_mpc_state();
                reset_nd_state();
                perf_metrics.reset();

                cout << "[JAKA] 视觉伺服: START" << endl;
            }
        }

            if (start_servo) {
                if (pixel_error > CONVERGENCE_RMS_PX) {
                    vpDisplay::displayText(I, 20, 20, "MPC ONLY MOVING", vpColor::cyan);
                } else {
                    vpDisplay::displayText(I, 20, 20, "Status: CONVERGED", vpColor::green);
                }
                vpDisplay::displayText(I, 40, 20, "Error: " + to_string(pixel_error), vpColor::yellow);
            } else {
                vpDisplay::displayText(I, 20, 20, "MPC ONLY paused: click to start", vpColor::orange);
            }
            vpDisplay::displayText(I, 60, 20,
                string("MPC: ") + (mpc_ok ? "OK" : "FOV fallback") +
                " | boundary=" + to_string(boundary_dist),
                mpc_ok ? vpColor::green : vpColor::red);

            // 与第二份代码一致：检测到标签后按帧绘制两组曲线。
            // 保留第一份代码的像素特征误差定义；相机速度显示最终视觉线程指令。
            plotter_mpc->plot(0, iter_plot_mpc, feature_error);
            plotter_mpc->plot(1, iter_plot_mpc, raw_v_cam);
            ++iter_plot_mpc;
        }
        else
        {
            g_tag_detected = false;
            reset_nd_state();
            reset_mpc_state();
            {
                lock_guard<mutex> lock(g_vis_mutex);
                g_latest_vcam = 0;
                g_latest_error = std::numeric_limits<double>::infinity();
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
    // 程序运行期间过滤指定的 OsqpEigen std::cerr 提示。
    // 其他 std::cerr 输出保持可见；结束前恢复原缓冲区。
    std::streambuf* original_cerr_buf = std::cerr.rdbuf();
    {
        OsqpWarningFilterBuf warning_filter(original_cerr_buf);
        std::cerr.rdbuf(&warning_filter);
        main_visual_servo();
        std::cerr.rdbuf(original_cerr_buf);
    }
    return 0;
}
