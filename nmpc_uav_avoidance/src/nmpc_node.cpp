/**
 * @file nmpc_node.cpp
 * @brief C++ NMPC UAV — Crazyflie (Crazyswarm2).
 *        SOFT dynamic obstacle + HARD SFC corridor (ALM).
 *
 *  RE-ADDED: SFC half-space corridor as a HARD constraint (ALM in the solver).
 *  Obstacle stays SOFT. Output stays cmd_full_state (carrot, clamped to the
 *  reference band per axis).
 *
 *  IMPORTANT LIMITATION: SFC constrains the SOLVER's predicted trajectory, not
 *  the executed command. The onboard PID tracks the clamped carrot and is
 *  unaware of the corridor, so it can cut corners across a half-space. Use the
 *  /nmpc/sfc_violation and /nmpc/pred_sfc_violation topics to monitor.
 *
 *  Coordinate frame: ENU (z-up). Signs from the working Crazyflie node:
 *     state_[6] =  euler.roll ;  state_[7] = -euler.pitch ; yaw_rate = -K_psi*yaw
 *
 *  Parameter layout (1548 total):
 *    [0:8] x0 | [8:11] u_prev | [11:101] p_ref(30*3)
 *    [101:104] p_obs | [104:107] v_obs | [107] r_obs
 *    [108:1548] SFC planes: step j plane m at 108 + (j*M_MAX+m)*4 = [nx,ny,nz,b]
 */

#include "nmpc_uav_avoidance/nmpc_node.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <functional>
#include <limits>

namespace nmpc_uav {

// ============================================================================
//  Free helpers
// ============================================================================

EulerAngles quaternion_to_euler(double qx, double qy, double qz, double qw) {
    EulerAngles e;
    double sinr = 2.0 * (qw * qx + qy * qz);
    double cosr = 1.0 - 2.0 * (qx * qx + qy * qy);
    e.roll = std::atan2(sinr, cosr);

    double sinp = 2.0 * (qw * qy - qz * qx);
    sinp = std::clamp(sinp, -1.0, 1.0);
    e.pitch = std::asin(sinp);

    double siny = 2.0 * (qw * qz + qx * qy);
    double cosy = 1.0 - 2.0 * (qy * qy + qz * qz);
    e.yaw = std::atan2(siny, cosy);
    return e;
}

Quaternion euler_to_quaternion(double roll, double pitch, double yaw) {
    double cr = std::cos(roll/2.0),  sr = std::sin(roll/2.0);
    double cp = std::cos(pitch/2.0), sp = std::sin(pitch/2.0);
    double cy = std::cos(yaw/2.0),   sy = std::sin(yaw/2.0);
    return {
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
        cr*cp*cy + sr*sp*sy
    };
}

std::array<double, NX> dynamics_step(const std::array<double, NX>& x,
                                     const std::array<double, NU>& u) {
    double px=x[0], py=x[1], pz=x[2];
    double vx=x[3], vy=x[4], vz=x[5];
    double phi=x[6], theta=x[7];
    double T_t=u[0], phi_ref=u[1], theta_ref=u[2];

    double ax =  T_t * std::cos(phi) * std::sin(theta);
    double ay = -T_t * std::sin(phi);
    double az =  T_t * std::cos(phi) * std::cos(theta) - GRAVITY;

    return {
        px + Ts*vx,
        py + Ts*vy,
        pz + Ts*vz,
        vx + Ts*(ax - AX*vx),
        vy + Ts*(ay - AY*vy),
        vz + Ts*(az - AZ*vz),
        phi   + (Ts/TAU_PHI)  *(K_PHI  *phi_ref   - phi),
        theta + (Ts/TAU_THETA)*(K_THETA*theta_ref - theta)
    };
}

// ============================================================================
//  Constructor / destructor
// ============================================================================

NMPCNode::NMPCNode() : Node("nmpc_node") {
    this->declare_parameter<std::string>("cf_name",     "cf231");
    this->declare_parameter<double>("obs_x",      1.0);
    this->declare_parameter<double>("obs_y",      0.0);
    this->declare_parameter<double>("obs_z",      1.0);
    this->declare_parameter<double>("obs_vx",     0.0);
    this->declare_parameter<double>("obs_vy",     0.0);
    this->declare_parameter<double>("obs_vz",     0.0);
    this->declare_parameter<double>("r_obs",      0.3);
    this->declare_parameter<double>("yaw_gain",   1.0);
    this->declare_parameter<double>("takeoff_height",   0.25);
    this->declare_parameter<double>("takeoff_duration", 3.0);
    this->declare_parameter<double>("hover_duration",   3.0);
    this->declare_parameter<int>("prediction_lookahead_steps", 5);

    this->declare_parameter<bool>("use_adaptive_lookahead", true);
    this->declare_parameter<double>("reference_speed",        0.10);
    this->declare_parameter<double>("lookahead_distance",     0.12);
    this->declare_parameter<double>("max_ref_ahead_distance", 0.30);
    this->declare_parameter<double>("carrot_max_dev_xy",      0.50);
    this->declare_parameter<double>("carrot_max_dev_z",       0.15);

    this->declare_parameter<int>("max_nonconv_streak", 1);

    cf_name_                    = this->get_parameter("cf_name").as_string();
    obs_pos_[0]                 = this->get_parameter("obs_x").as_double();
    obs_pos_[1]                 = this->get_parameter("obs_y").as_double();
    obs_pos_[2]                 = this->get_parameter("obs_z").as_double();
    obs_vel_[0]                 = this->get_parameter("obs_vx").as_double();
    obs_vel_[1]                 = this->get_parameter("obs_vy").as_double();
    obs_vel_[2]                 = this->get_parameter("obs_vz").as_double();
    r_obs_                      = this->get_parameter("r_obs").as_double();
    K_psi_                      = this->get_parameter("yaw_gain").as_double();
    takeoff_height_             = this->get_parameter("takeoff_height").as_double();
    takeoff_duration_           = this->get_parameter("takeoff_duration").as_double();
    hover_duration_             = this->get_parameter("hover_duration").as_double();
    prediction_lookahead_steps_ = this->get_parameter("prediction_lookahead_steps").as_int();
    prediction_lookahead_steps_ = std::clamp(prediction_lookahead_steps_, 1, N);

    use_adaptive_lookahead_     = this->get_parameter("use_adaptive_lookahead").as_bool();
    reference_speed_            = this->get_parameter("reference_speed").as_double();
    lookahead_distance_         = this->get_parameter("lookahead_distance").as_double();
    max_ref_ahead_distance_     = this->get_parameter("max_ref_ahead_distance").as_double();
    carrot_max_dev_xy_          = this->get_parameter("carrot_max_dev_xy").as_double();
    carrot_max_dev_z_           = this->get_parameter("carrot_max_dev_z").as_double();

    max_nonconv_streak_ = this->get_parameter("max_nonconv_streak").as_int();
    max_nonconv_streak_ = std::max(1, max_nonconv_streak_);

    reference_speed_        = std::max(0.0, reference_speed_);
    lookahead_distance_     = std::max(0.0, lookahead_distance_);
    max_ref_ahead_distance_ = std::max(lookahead_distance_, max_ref_ahead_distance_);
    carrot_max_dev_xy_      = std::max(0.0, carrot_max_dev_xy_);
    carrot_max_dev_z_       = std::max(0.0, carrot_max_dev_z_);

    RCLCPP_INFO(this->get_logger(), "Initialising OpEn solver (C bindings)...");
    solver_ = std::make_unique<OpEnSolver>();
    solver_->start();
    RCLCPP_INFO(this->get_logger(),
        "OpEn solver ready. N_PARAMS=%d M_MAX=%d", N_PARAMS, M_MAX);

    const std::string odom_topic           = "/" + cf_name_ + "/odom";
    const std::string cmd_full_state_topic = "/" + cf_name_ + "/cmd_full_state";
    const std::string takeoff_srv          = "/" + cf_name_ + "/takeoff";
    const std::string land_srv             = "/" + cf_name_ + "/land";

    // ---- Subscribers ----
    sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic, 10,
        std::bind(&NMPCNode::odom_callback, this, std::placeholders::_1));
    sub_obs_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/obstacle/odom", 10,
        std::bind(&NMPCNode::obs_odom_callback, this, std::placeholders::_1));
    sub_path_ = this->create_subscription<nav_msgs::msg::Path>(
        "/planned_path", 10,
        std::bind(&NMPCNode::path_callback, this, std::placeholders::_1));
    sub_sfc_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        "/sfc_coefficients", 10,
        std::bind(&NMPCNode::sfc_callback, this, std::placeholders::_1));

    // ---- Publishers ----
    pub_cmd_full_state_ = this->create_publisher<crazyflie_interfaces::msg::FullState>(
        cmd_full_state_topic, 10);
    pub_pred_path_      = this->create_publisher<nav_msgs::msg::Path>(
        "/nmpc/predicted_path", 10);
    pub_obs_marker_     = this->create_publisher<visualization_msgs::msg::Marker>(
        "/nmpc/obstacle_marker", 10);
    pub_solver_time_    = this->create_publisher<std_msgs::msg::Float64>(
        "/nmpc/solver_time_ms", 10);
    pub_cost_function_  = this->create_publisher<std_msgs::msg::Float64>(
        "/nmpc/cost_function", 10);
    pub_control_input_  = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "/nmpc/control_input", 10);
    pub_jerk_           = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "/nmpc/jerk", 10);
    pub_obs_viol_       = this->create_publisher<std_msgs::msg::Float64>(
        "/nmpc/obs_violation", 10);
    pub_obs_dist_       = this->create_publisher<std_msgs::msg::Float64>(
        "/nmpc/obs_distance", 10);
    pub_pred_obs_viol_  = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "/nmpc/pred_obs_violation", 10);
    pub_sfc_viol_       = this->create_publisher<std_msgs::msg::Float64>(
        "/nmpc/sfc_violation", 10);
    pub_sfc_margin_     = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "/nmpc/sfc_margin", 10);
    pub_pred_sfc_viol_  = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "/nmpc/pred_sfc_violation", 10);

    // ---- Service clients ----
    cli_takeoff_ = this->create_client<crazyflie_interfaces::srv::Takeoff>(takeoff_srv);
    cli_land_    = this->create_client<crazyflie_interfaces::srv::Land>(land_srv);

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(Ts * 1000)),
        std::bind(&NMPCNode::control_loop, this));

    RCLCPP_INFO(this->get_logger(),
        "NMPC node ready for '%s' (ENU, %d params, SOFT obstacle + HARD SFC, "
        "M_MAX=%d) | OUT=cmd_full_state (k=%d) | ref_v=%.3f la=%.2f maxahead=%.2f "
        "| carrot_dev xy=%.2f z=%.2f | takeoff=%.2fm/%.1fs hover=%.1fs",
        cf_name_.c_str(), N_PARAMS, M_MAX, prediction_lookahead_steps_,
        reference_speed_, lookahead_distance_, max_ref_ahead_distance_,
        carrot_max_dev_xy_, carrot_max_dev_z_,
        takeoff_height_, takeoff_duration_, hover_duration_);
}

NMPCNode::~NMPCNode() {
    RCLCPP_INFO(this->get_logger(), "Requesting landing and shutting down...");
    request_land();
    if (solver_) solver_->kill_solver();
}

// ============================================================================
//  Callbacks
// ============================================================================

void NMPCNode::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    auto& p = msg->pose.pose.position;
    auto& v = msg->twist.twist.linear;
    auto& q = msg->pose.pose.orientation;
    EulerAngles euler = quaternion_to_euler(q.x, q.y, q.z, q.w);

    state_[0] = p.x;  state_[1] = p.y;  state_[2] = p.z;
    state_[3] = v.x;  state_[4] = v.y;  state_[5] = v.z;
    state_[6] =  euler.roll;     // ENU sign convention (NOT negated)
    state_[7] = -euler.pitch;
    yaw_            = euler.yaw;
    state_received_ = true;
}

void NMPCNode::obs_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    obs_pos_[0] = msg->pose.pose.position.x;
    obs_pos_[1] = msg->pose.pose.position.y;
    obs_pos_[2] = msg->pose.pose.position.z;
    obs_vel_[0] = msg->twist.twist.linear.x;
    obs_vel_[1] = msg->twist.twist.linear.y;
    obs_vel_[2] = msg->twist.twist.linear.z;
    obs_received_ = true;
}

void NMPCNode::path_callback(const nav_msgs::msg::Path::SharedPtr msg) {
    path_points_.clear();
    path_points_.reserve(msg->poses.size());
    for (const auto& ps : msg->poses) {
        path_points_.push_back({
            ps.pose.position.x,
            ps.pose.position.y,
            ps.pose.position.z
        });
    }

    rebuild_path_arclength();

    if (!path_points_.empty()) {
        std::array<double, 3> p_now = {state_[0], state_[1], state_[2]};
        size_t nearest_idx = 0;
        double nearest_s = state_received_
            ? find_nearest_path_s(p_now, &nearest_idx) : 0.0;
        ref_progress_s_ = std::clamp(nearest_s, 0.0, path_length_);
        path_idx_ = nearest_idx;
        ref_progress_initialized_ = true;
    } else {
        path_idx_ = 0;
        ref_progress_s_ = 0.0;
        ref_progress_initialized_ = false;
    }

    last_solve_time_s_ = Ts;
    RCLCPP_INFO(this->get_logger(),
        "Received path with %zu waypoints. path_length=%.3f m, ref_s=%.3f",
        path_points_.size(), path_length_, ref_progress_s_);
}

void NMPCNode::sfc_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    const auto& d = msg->data;
    if (d.empty()) {
        RCLCPP_WARN(this->get_logger(), "Empty SFC message.");
        return;
    }

    std::vector<SfcPolytope> sfcs;
    size_t idx = 0;
    int n_poly = static_cast<int>(d[idx++]);
    sfcs.reserve(std::max(0, n_poly));

    for (int i = 0; i < n_poly; ++i) {
        if (idx + 2 > d.size()) {
            RCLCPP_WARN(this->get_logger(), "SFC msg truncated at header %d.", i);
            break;
        }
        SfcPolytope poly;
        poly.segment_index = static_cast<int>(d[idx++]);
        int M = static_cast<int>(d[idx++]);
        if (M < 0 || idx + 4*static_cast<size_t>(M) > d.size()) {
            RCLCPP_WARN(this->get_logger(), "SFC msg malformed at poly %d (M=%d).",
                        i, M);
            break;
        }
        poly.planes.reserve(M);
        for (int m = 0; m < M; ++m) {
            SfcPlane pl;
            pl.n = { d[idx], d[idx+1], d[idx+2] };
            pl.b = d[idx+3];
            idx += 4;
            poly.planes.push_back(pl);
        }
        sfcs.push_back(std::move(poly));
    }

    sfcs_          = std::move(sfcs);
    sfc_received_  = !sfcs_.empty();
    last_poly_idx_ = 0;

    RCLCPP_INFO(this->get_logger(), "Received %zu SFC polytopes.", sfcs_.size());
}

// ============================================================================
//  Crazyflie takeoff / land services
// ============================================================================

void NMPCNode::request_takeoff() {
    if (!cli_takeoff_->wait_for_service(std::chrono::milliseconds(2000))) {
        RCLCPP_ERROR(this->get_logger(),
            "Takeoff service '/%s/takeoff' not available.", cf_name_.c_str());
        return;
    }
    auto req = std::make_shared<crazyflie_interfaces::srv::Takeoff::Request>();
    req->group_mask = 0;
    req->height     = takeoff_height_;
    double whole_sec = std::floor(takeoff_duration_);
    req->duration.sec     = static_cast<int32_t>(whole_sec);
    req->duration.nanosec = static_cast<uint32_t>(
        (takeoff_duration_ - whole_sec) * 1e9);
    cli_takeoff_->async_send_request(req);
    RCLCPP_INFO(this->get_logger(),
        "Takeoff requested (height=%.2fm, duration=%.2fs). "
        "Hover %.1fs before NMPC engages.",
        takeoff_height_, takeoff_duration_, hover_duration_);
}

void NMPCNode::request_land() {
    if (!cli_land_ || !cli_land_->wait_for_service(std::chrono::milliseconds(500))) {
        RCLCPP_WARN(this->get_logger(), "Land service not available.");
        return;
    }
    auto req = std::make_shared<crazyflie_interfaces::srv::Land::Request>();
    req->group_mask = 0;
    req->height     = 0.0;
    req->duration.sec     = 3;
    req->duration.nanosec = 0;
    cli_land_->async_send_request(req);
    RCLCPP_INFO(this->get_logger(), "Land requested.");
}

// ============================================================================
//  Arc-length reference: constant-speed virtual target
// ============================================================================

void NMPCNode::rebuild_path_arclength() {
    path_s_.clear();
    path_length_ = 0.0;
    if (path_points_.empty()) return;

    path_s_.resize(path_points_.size(), 0.0);
    for (size_t i = 1; i < path_points_.size(); ++i) {
        const double dx = path_points_[i][0] - path_points_[i - 1][0];
        const double dy = path_points_[i][1] - path_points_[i - 1][1];
        const double dz = path_points_[i][2] - path_points_[i - 1][2];
        path_length_ += std::sqrt(dx*dx + dy*dy + dz*dz);
        path_s_[i] = path_length_;
    }
}

size_t NMPCNode::find_segment_index_for_s(double s) const {
    if (path_points_.size() < 2 || path_s_.empty()) return 0;
    s = std::clamp(s, 0.0, path_length_);
    auto it = std::upper_bound(path_s_.begin(), path_s_.end(), s);
    if (it == path_s_.begin()) return 0;
    size_t idx = static_cast<size_t>(std::distance(path_s_.begin(), it) - 1);
    if (idx >= path_points_.size() - 1) idx = path_points_.size() - 2;
    return idx;
}

std::array<double, 3> NMPCNode::sample_path_at_s(double s) const {
    if (path_points_.empty()) return {state_[0], state_[1], state_[2]};
    if (path_points_.size() == 1 || path_length_ <= 1e-9 ||
        path_s_.size() != path_points_.size()) {
        return path_points_.front();
    }
    s = std::clamp(s, 0.0, path_length_);
    const size_t i = find_segment_index_for_s(s);
    const double s0 = path_s_[i];
    const double s1 = path_s_[i + 1];
    const double denom = std::max(s1 - s0, 1e-9);
    const double a = std::clamp((s - s0) / denom, 0.0, 1.0);
    return {
        (1.0 - a) * path_points_[i][0] + a * path_points_[i + 1][0],
        (1.0 - a) * path_points_[i][1] + a * path_points_[i + 1][1],
        (1.0 - a) * path_points_[i][2] + a * path_points_[i + 1][2]
    };
}

double NMPCNode::find_nearest_path_s(const std::array<double, 3>& p,
                                     size_t* nearest_idx) const {
    if (path_points_.empty()) {
        if (nearest_idx) *nearest_idx = 0;
        return 0.0;
    }
    if (path_points_.size() == 1 || path_s_.size() != path_points_.size()) {
        if (nearest_idx) *nearest_idx = 0;
        return 0.0;
    }

    double best_d2 = std::numeric_limits<double>::infinity();
    double best_s = 0.0;
    size_t best_idx = 0;

    for (size_t i = 0; i + 1 < path_points_.size(); ++i) {
        const auto& a = path_points_[i];
        const auto& b = path_points_[i + 1];
        const double abx = b[0] - a[0];
        const double aby = b[1] - a[1];
        const double abz = b[2] - a[2];
        const double apx = p[0] - a[0];
        const double apy = p[1] - a[1];
        const double apz = p[2] - a[2];
        const double ab2 = abx*abx + aby*aby + abz*abz;

        double t = 0.0;
        if (ab2 > 1e-12) {
            t = std::clamp((apx*abx + apy*aby + apz*abz) / ab2, 0.0, 1.0);
        }
        const double qx = a[0] + t * abx;
        const double qy = a[1] + t * aby;
        const double qz = a[2] + t * abz;
        const double dx = p[0] - qx;
        const double dy = p[1] - qy;
        const double dz = p[2] - qz;
        const double d2 = dx*dx + dy*dy + dz*dz;

        if (d2 < best_d2) {
            best_d2 = d2;
            best_s = path_s_[i] + t * (path_s_[i + 1] - path_s_[i]);
            best_idx = (t > 0.5) ? i + 1 : i;
        }
    }

    if (nearest_idx) *nearest_idx = best_idx;
    return std::clamp(best_s, 0.0, path_length_);
}

void NMPCNode::update_reference() {
    if (path_points_.empty()) return;
    if (path_s_.size() != path_points_.size()) rebuild_path_arclength();

    std::array<double, 3> p_now = {state_[0], state_[1], state_[2]};
    size_t nearest_idx = path_idx_;
    const double nearest_s = find_nearest_path_s(p_now, &nearest_idx);

    if (!ref_progress_initialized_) {
        ref_progress_s_ = nearest_s;
        ref_progress_initialized_ = true;
    }

    if (use_adaptive_lookahead_) {
        const double ds = reference_speed_ * Ts;
        ref_progress_s_ += ds;
        if (ref_progress_s_ < nearest_s) ref_progress_s_ = nearest_s;
        const double max_ahead_s = nearest_s + max_ref_ahead_distance_;
        ref_progress_s_ = std::clamp(ref_progress_s_, nearest_s, max_ahead_s);
    } else {
        ref_progress_s_ = nearest_s;
    }

    ref_progress_s_ = std::clamp(ref_progress_s_, 0.0, path_length_);
    const double first_ref_s = std::clamp(ref_progress_s_ + lookahead_distance_,
                                          0.0, path_length_);
    path_idx_ = find_segment_index_for_s(first_ref_s);

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "REF: nearest_s=%.3f ref_s=%.3f first_ref_s=%.3f v=%.3f idx=%zu/%zu poly=%d",
        nearest_s, ref_progress_s_, first_ref_s, reference_speed_,
        path_idx_, path_points_.size(), last_poly_idx_);
}

// ============================================================================
//  SFC: polytope assignment + parameter packing
// ============================================================================

bool NMPCNode::point_inside_polytope(const std::array<double,3>& p,
                                     const std::vector<SfcPlane>& planes,
                                     double tol) const {
    for (const auto& pl : planes) {
        double v = pl.n[0]*p[0] + pl.n[1]*p[1] + pl.n[2]*p[2] - pl.b;
        if (v > tol) return false;
    }
    return true;
}

int NMPCNode::find_closest_polytope(const std::array<double,3>& p) const {
    // No segment endpoints transmitted: pick the polytope whose worst plane
    // violation at p is smallest.
    int best = 0;
    double best_v = std::numeric_limits<double>::max();
    for (size_t i = 0; i < sfcs_.size(); ++i) {
        double worst = -std::numeric_limits<double>::infinity();
        for (const auto& pl : sfcs_[i].planes) {
            double v = pl.n[0]*p[0] + pl.n[1]*p[1] + pl.n[2]*p[2] - pl.b;
            if (v > worst) worst = v;
        }
        if (worst < best_v) { best_v = worst; best = static_cast<int>(i); }
    }
    return best;
}

int NMPCNode::find_polytope_for_point(const std::array<double,3>& p,
                                      int idx_start) const {
    if (sfcs_.empty()) return -1;
    const int Np = static_cast<int>(sfcs_.size());
    idx_start = std::clamp(idx_start, 0, Np - 1);

    for (int j = idx_start; j < Np; ++j) {
        if (point_inside_polytope(p, sfcs_[j].planes)) return j;
    }
    for (int j = idx_start - 1; j >= 0; --j) {
        if (point_inside_polytope(p, sfcs_[j].planes)) return j;
    }
    return find_closest_polytope(p);
}

std::vector<int> NMPCNode::assign_polytopes_to_horizon() {
    std::vector<int> assignment(N, -1);
    if (sfcs_.empty() || path_points_.empty()) return assignment;

    std::array<double,3> cur_pos = {state_[0], state_[1], state_[2]};
    int idx = find_polytope_for_point(cur_pos, last_poly_idx_);
    if (idx < 0) idx = 0;
    last_poly_idx_ = idx;

    const double first_ref_s = std::clamp(ref_progress_s_ + lookahead_distance_,
                                          0.0, path_length_);
    const double ref_ds = std::max(reference_speed_ * Ts, 0.002);

    for (int j = 0; j < N; ++j) {
        const double s_j = std::clamp(first_ref_s + j * ref_ds, 0.0, path_length_);
        const auto p_ref = sample_path_at_s(s_j);
        int new_idx = find_polytope_for_point(p_ref, idx);
        if (new_idx < 0) new_idx = idx;
        assignment[j] = new_idx;
        idx = new_idx;
    }
    return assignment;
}

void NMPCNode::fill_sfc_params(std::vector<double>& params,
                               const std::vector<int>& assignment) const {
    if (sfcs_.empty()) return;

    for (int j = 0; j < N; ++j) {
        int p_idx = assignment[j];
        if (p_idx < 0 || p_idx >= static_cast<int>(sfcs_.size())) continue;
        const auto& planes = sfcs_[p_idx].planes;

        const int M = std::min(static_cast<int>(planes.size()), M_MAX);
        for (int m = 0; m < M; ++m) {
            const int off = SFC_START + (j * M_MAX + m) * 4;
            params[off + 0] = planes[m].n[0];
            params[off + 1] = planes[m].n[1];
            params[off + 2] = planes[m].n[2];
            params[off + 3] = planes[m].b;
        }
        // Inactive planes (m >= M) remain zero -> g = 0 <= 0 trivially.
    }
}

// ============================================================================
//  Diagnostics (PlotJuggler)
// ============================================================================

void NMPCNode::publish_obstacle_diagnostics(const std::vector<double>& z_star) {
    {
        const double dx = state_[0] - obs_pos_[0];
        const double dy = state_[1] - obs_pos_[1];
        const double dz = state_[2] - obs_pos_[2];
        const double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        std_msgs::msg::Float64 m;
        m.data = dist;          pub_obs_dist_->publish(m);
        m.data = r_obs_ - dist; pub_obs_viol_->publish(m);
    }

    std_msgs::msg::Float64MultiArray pred;
    pred.data.reserve(N + 1);

    std::array<double, NX> x_k = state_;
    for (int j = 0; j <= N; ++j) {
        const double r_s_j = R_S_MAX * static_cast<double>(j) / static_cast<double>(N);
        const double ox = obs_pos_[0] + j * Ts * obs_vel_[0];
        const double oy = obs_pos_[1] + j * Ts * obs_vel_[1];
        const double oz = obs_pos_[2] + j * Ts * obs_vel_[2];
        const double dx = x_k[0] - ox;
        const double dy = x_k[1] - oy;
        const double dz = x_k[2] - oz;
        const double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        pred.data.push_back((r_obs_ + r_s_j) - dist);

        if (j < N) {
            int idx_u = j * NU;
            if (idx_u + NU <= static_cast<int>(z_star.size())) {
                std::array<double, NU> uj = {
                    z_star[idx_u], z_star[idx_u+1], z_star[idx_u+2]
                };
                x_k = dynamics_step(x_k, uj);
            } else {
                break;
            }
        }
    }
    pub_pred_obs_viol_->publish(pred);
}

void NMPCNode::publish_sfc_diagnostics(const std::vector<double>& z_star,
                                       const std::vector<int>& assignment) {
    // ---- CURRENT SFC margin/violation for the drone's polytope ----
    // margin_m = b_m - n_m . p (positive = inside); violation = max_m(-margin_m)
    {
        const std::array<double,3> p_now = {state_[0], state_[1], state_[2]};
        std_msgs::msg::Float64MultiArray margins;
        double worst = -std::numeric_limits<double>::infinity();

        if (!sfcs_.empty()) {
            int idx = (last_poly_idx_ >= 0 &&
                       last_poly_idx_ < static_cast<int>(sfcs_.size()))
                          ? last_poly_idx_ : 0;
            const auto& planes = sfcs_[idx].planes;
            margins.data.reserve(planes.size());
            for (const auto& pl : planes) {
                double margin = pl.b - (pl.n[0]*p_now[0]
                                       + pl.n[1]*p_now[1]
                                       + pl.n[2]*p_now[2]);
                margins.data.push_back(margin);
                if (-margin > worst) worst = -margin;
            }
        }
        if (worst == -std::numeric_limits<double>::infinity()) worst = 0.0;

        pub_sfc_margin_->publish(margins);
        std_msgs::msg::Float64 m; m.data = worst;  // >0 => outside corridor
        pub_sfc_viol_->publish(m);
    }

    // ---- PREDICTED SFC violation over horizon ----
    std_msgs::msg::Float64MultiArray pred_sfc;
    pred_sfc.data.reserve(N + 1);

    std::array<double, NX> x_k = state_;
    for (int j = 0; j <= N; ++j) {
        double worst_sfc = 0.0;
        if (!sfcs_.empty() && !assignment.empty()) {
            int j_use = std::min(j, N - 1);
            int p_idx = (j_use < static_cast<int>(assignment.size()))
                            ? assignment[j_use] : -1;
            if (p_idx >= 0 && p_idx < static_cast<int>(sfcs_.size())) {
                const auto& planes = sfcs_[p_idx].planes;
                double worst = -std::numeric_limits<double>::infinity();
                for (const auto& pl : planes) {
                    double v = pl.n[0]*x_k[0] + pl.n[1]*x_k[1]
                             + pl.n[2]*x_k[2] - pl.b;
                    if (v > worst) worst = v;
                }
                if (worst != -std::numeric_limits<double>::infinity())
                    worst_sfc = worst;
            }
        }
        pred_sfc.data.push_back(worst_sfc);  // >0 => predicted outside corridor

        if (j < N) {
            int idx_u = j * NU;
            if (idx_u + NU <= static_cast<int>(z_star.size())) {
                std::array<double, NU> uj = {
                    z_star[idx_u], z_star[idx_u+1], z_star[idx_u+2]
                };
                x_k = dynamics_step(x_k, uj);
            } else {
                break;
            }
        }
    }
    pub_pred_sfc_viol_->publish(pred_sfc);
}

// ============================================================================
//  Main control loop (20 Hz)
// ============================================================================

void NMPCNode::control_loop() {
    if (!state_received_) return;

    // ---- Takeoff state machine ----
    if (!takeoff_requested_) {
        request_takeoff();
        takeoff_requested_ = true;
        return;
    }
    if (!takeoff_done_) {
        settle_counter_++;
        double elapsed    = settle_counter_ * Ts;
        double total_wait = takeoff_duration_ + hover_duration_;
        if (elapsed >= total_wait) {
            takeoff_done_ = true;
            u_prev_ = {GRAVITY, 0.0, 0.0};
            RCLCPP_INFO(this->get_logger(),
                "Takeoff + hover complete (%.1fs elapsed), starting NMPC.", elapsed);
        } else {
            return;
        }
    }

    update_reference();

    // ---- Build parameter vector (1548) ----
    std::vector<double> params(N_PARAMS, 0.0);
    for (int i = 0; i < NX; ++i) params[i]     = state_[i];
    for (int i = 0; i < NU; ++i) params[8 + i] = u_prev_[i];

    const int ref_start = 11;
    const double first_ref_s = std::clamp(ref_progress_s_ + lookahead_distance_,
                                          0.0, path_length_);
    const double ref_ds = std::max(reference_speed_ * Ts, 0.002);

    for (int j = 0; j < N; ++j) {
        if (!path_points_.empty()) {
            const double s_j = std::clamp(first_ref_s + j * ref_ds, 0.0, path_length_);
            const auto p_ref = sample_path_at_s(s_j);
            params[ref_start + j*3 + 0] = p_ref[0];
            params[ref_start + j*3 + 1] = p_ref[1];
            params[ref_start + j*3 + 2] = p_ref[2];
        } else {
            params[ref_start + j*3 + 0] = state_[0];
            params[ref_start + j*3 + 1] = state_[1];
            params[ref_start + j*3 + 2] = state_[2];
        }
    }

    params[101] = obs_pos_[0];
    params[102] = obs_pos_[1];
    params[103] = obs_pos_[2];
    params[104] = obs_vel_[0];
    params[105] = obs_vel_[1];
    params[106] = obs_vel_[2];
    params[107] = r_obs_;

    // ---- SFC planes (zero-pad inactive) ----
    auto poly_assignment = assign_polytopes_to_horizon();
    fill_sfc_params(params, poly_assignment);

    if (!obs_received_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "Obstacle odom not yet received; using default obstacle params.");
    }
    if (!sfc_received_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "SFC not yet received; flying without corridor constraints.");
    }

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "obs=[%.2f,%.2f,%.2f] | UAV=[%.2f,%.2f,%.2f] idx=%zu/%zu poly=%d",
        obs_pos_[0], obs_pos_[1], obs_pos_[2],
        state_[0], state_[1], state_[2],
        path_idx_, path_points_.size(), last_poly_idx_);

    // ---- Solve ----
    OpEnResult result;
    try {
        result = solver_->solve(params);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Solver call failed: %s", e.what());
        publish_fallback_setpoint();
        return;
    }

    const bool converged =
        result.ok && (result.exit_status.find("Converged") != std::string::npos);

    if (!result.ok || result.solution.size() < static_cast<size_t>(NU)) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "Solver hard error (%s); holding position.",
            result.exit_status.c_str());
        publish_obstacle_diagnostics({});
        publish_sfc_diagnostics({}, poly_assignment);
        publish_fallback_setpoint();
        return;
    }
    if (!converged) {
        // Option A: NEVER send a non-converged (possibly saturated) solution to
        // the airframe. A non-converged ALM result is typically the cause of the
        // T=5/13.5, phi/theta=+/-0.35 saturated garbage that flung the drone into
        // the wall. Hold position instead; resume when the solver recovers.
        nonconv_streak_++;
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "Solver did not converge (%s) — holding position (streak=%d). "
            "Likely tight SFC/obstacle intersection.",
            result.exit_status.c_str(), nonconv_streak_);
        publish_obstacle_diagnostics(result.solution);
        publish_sfc_diagnostics(result.solution, poly_assignment);
        publish_fallback_setpoint();
        return;
    }
    nonconv_streak_ = 0;

    const auto& z_star = result.solution;

    last_solve_time_s_ = result.solve_time_ms * 1e-3;

    double T_opt         = z_star[0];
    double phi_ref_opt   = z_star[1];
    double theta_ref_opt = z_star[2];

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "SOLVE: T=%.3f phi=%.6f theta=%.6f | %s | %.2fms",
        T_opt, phi_ref_opt, theta_ref_opt,
        result.exit_status.c_str(), result.solve_time_ms);

    // ---- Telemetry ----
    {
        std_msgs::msg::Float64 m; m.data = result.solve_time_ms;
        pub_solver_time_->publish(m);
    }
    {
        std_msgs::msg::Float64 m; m.data = result.cost;
        pub_cost_function_->publish(m);
    }
    {
        std_msgs::msg::Float64MultiArray m;
        m.data = {T_opt, phi_ref_opt, theta_ref_opt};
        pub_control_input_->publish(m);
    }
    {
        std_msgs::msg::Float64MultiArray m;
        if (z_star.size() >= static_cast<size_t>(2 * NU)) {
            std::array<double, NX> x0 = state_;
            std::array<double, NU> u0 = {z_star[0],  z_star[1],    z_star[2]};
            std::array<double, NU> u1 = {z_star[NU], z_star[NU+1], z_star[NU+2]};
            std::array<double, NX> x1 = dynamics_step(x0, u0);
            std::array<double, NX> x2 = dynamics_step(x1, u1);
            const double ax0 = (x1[3] - x0[3]) / Ts;
            const double ay0 = (x1[4] - x0[4]) / Ts;
            const double az0 = (x1[5] - x0[5]) / Ts;
            const double ax1 = (x2[3] - x1[3]) / Ts;
            const double ay1 = (x2[4] - x1[4]) / Ts;
            const double az1 = (x2[5] - x1[5]) / Ts;
            m.data = {(ax1 - ax0) / Ts, (ay1 - ay0) / Ts, (az1 - az0) / Ts};
        } else {
            m.data = {0.0, 0.0, 0.0};
        }
        pub_jerk_->publish(m);
    }

    publish_obstacle_diagnostics(z_star);
    publish_sfc_diagnostics(z_star, poly_assignment);

    u_prev_ = {T_opt, phi_ref_opt, theta_ref_opt};

    // ---- Reference at step k = clamp centre for the carrot ----
    const int kc = std::min(prediction_lookahead_steps_, N - 1);
    std::array<double, 3> ref_k;
    if (path_points_.empty()) {
        ref_k = {state_[0], state_[1], state_[2]};
    } else {
        const double s_k = std::clamp(first_ref_s + kc * ref_ds, 0.0, path_length_);
        ref_k = sample_path_at_s(s_k);
    }

    // ---- Command out: k-step carrot (clamped) as FullState ----
    publish_full_state(state_, z_star, ref_k);
    publish_predicted_path(z_star);
    publish_obstacle_marker();
}

// ============================================================================
//  Crazyflie cmd_full_state publishing
// ============================================================================

void NMPCNode::publish_full_state(const std::array<double, NX>& x_now,
                                  const std::vector<double>& u_seq,
                                  const std::array<double, 3>& ref_clamp_center) {
    const int k = prediction_lookahead_steps_;

    auto u_at = [&](int i) -> std::array<double, NU> {
        const int base = i * NU;
        if (base + NU <= static_cast<int>(u_seq.size())) {
            return {u_seq[base], u_seq[base + 1], u_seq[base + 2]};
        }
        const int last = (static_cast<int>(u_seq.size()) / NU) - 1;
        if (last >= 0) {
            const int lb = last * NU;
            return {u_seq[lb], u_seq[lb + 1], u_seq[lb + 2]};
        }
        return u_prev_;
    };

    std::array<double, NX> x_k = x_now;
    for (int i = 0; i < k; ++i) x_k = dynamics_step(x_k, u_at(i));
    const std::array<double, NX> x_kp1 = dynamics_step(x_k, u_at(k));

    const double ax = (x_kp1[3] - x_k[3]) / Ts;
    const double ay = (x_kp1[4] - x_k[4]) / Ts;
    const double az = (x_kp1[5] - x_k[5]) / Ts;

    // Carrot position clamp (kills momentum-driven z runaway).
    std::array<double,3> cmd_pos = {
        std::clamp(x_k[0], ref_clamp_center[0] - carrot_max_dev_xy_,
                           ref_clamp_center[0] + carrot_max_dev_xy_),
        std::clamp(x_k[1], ref_clamp_center[1] - carrot_max_dev_xy_,
                           ref_clamp_center[1] + carrot_max_dev_xy_),
        std::clamp(x_k[2], ref_clamp_center[2] - carrot_max_dev_z_,
                           ref_clamp_center[2] + carrot_max_dev_z_)
    };
    std::array<double,3> cmd_vel = { x_k[3], x_k[4], x_k[5] };

    const double yaw_rate = -K_psi_ * yaw_;
    const Quaternion q = euler_to_quaternion(0.0, 0.0, 0.0);

    crazyflie_interfaces::msg::FullState msg;
    msg.header.stamp    = this->now();
    msg.header.frame_id = "world";

    msg.pose.position.x = cmd_pos[0];
    msg.pose.position.y = cmd_pos[1];
    msg.pose.position.z = cmd_pos[2];
    msg.pose.orientation.x = q.x;
    msg.pose.orientation.y = q.y;
    msg.pose.orientation.z = q.z;
    msg.pose.orientation.w = q.w;

    msg.twist.linear.x  = cmd_vel[0];
    msg.twist.linear.y  = cmd_vel[1];
    msg.twist.linear.z  = cmd_vel[2];
    msg.twist.angular.x = 0.0;
    msg.twist.angular.y = 0.0;
    msg.twist.angular.z = yaw_rate;

    msg.acc.x = ax;
    msg.acc.y = ay;
    msg.acc.z = az;

    pub_cmd_full_state_->publish(msg);
}

// ============================================================================
//  Fallback / visualisation
// ============================================================================

void NMPCNode::publish_fallback_setpoint() {
    const double yaw_rate = -K_psi_ * yaw_;
    const Quaternion q = euler_to_quaternion(0.0, 0.0, 0.0);

    crazyflie_interfaces::msg::FullState msg;
    msg.header.stamp    = this->now();
    msg.header.frame_id = "world";

    msg.pose.position.x = state_[0];
    msg.pose.position.y = state_[1];
    msg.pose.position.z = state_[2];
    msg.pose.orientation.x = q.x;
    msg.pose.orientation.y = q.y;
    msg.pose.orientation.z = q.z;
    msg.pose.orientation.w = q.w;

    msg.twist.linear.x  = 0.0;
    msg.twist.linear.y  = 0.0;
    msg.twist.linear.z  = 0.0;
    msg.twist.angular.x = 0.0;
    msg.twist.angular.y = 0.0;
    msg.twist.angular.z = yaw_rate;

    msg.acc.x = 0.0; msg.acc.y = 0.0; msg.acc.z = 0.0;

    pub_cmd_full_state_->publish(msg);
}

void NMPCNode::publish_predicted_path(const std::vector<double>& z_star) {
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp    = this->now();
    path_msg.header.frame_id = "world";

    std::array<double, NX> x_k = state_;
    for (int j = 0; j < N; ++j) {
        int idx = j * NU;
        if (idx + NU > static_cast<int>(z_star.size())) break;
        std::array<double, NU> uj = {z_star[idx], z_star[idx+1], z_star[idx+2]};
        x_k = dynamics_step(x_k, uj);

        geometry_msgs::msg::PoseStamped ps;
        ps.header = path_msg.header;
        ps.pose.position.x = x_k[0];
        ps.pose.position.y = x_k[1];
        ps.pose.position.z = x_k[2];
        path_msg.poses.push_back(ps);
    }
    pub_pred_path_->publish(path_msg);
}

void NMPCNode::publish_obstacle_marker() {
    visualization_msgs::msg::Marker m;
    m.header.stamp    = this->now();
    m.header.frame_id = "world";
    m.ns = "obstacle"; m.id = 0;
    m.type = visualization_msgs::msg::Marker::SPHERE;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.position.x = obs_pos_[0];
    m.pose.position.y = obs_pos_[1];
    m.pose.position.z = obs_pos_[2];
    m.pose.orientation.w = 1.0;
    double d = 2.0 * r_obs_;
    m.scale.x = d; m.scale.y = d; m.scale.z = d;
    m.color.r = 1.0f; m.color.g = 0.3f; m.color.b = 0.0f; m.color.a = 0.5f;
    pub_obs_marker_->publish(m);
}

}  // namespace nmpc_uav