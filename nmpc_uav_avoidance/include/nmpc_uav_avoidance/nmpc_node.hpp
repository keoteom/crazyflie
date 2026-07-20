/**
 * @file nmpc_node.hpp
 * @brief NMPC UAV — Crazyflie (Crazyswarm2).
 *        SOFT dynamic obstacle + HARD SFC corridor (ALM).
 *
 *  Output: cmd_full_state carrot (clamped to reference band per axis).
 *  NOTE: SFC bounds the SOLVER's predicted trajectory, NOT the executed
 *  command (the onboard PID tracks the clamped carrot and is unaware of SFC).
 *
 *  Parameter layout (1548 total):
 *    [0:8]      x0
 *    [8:11]     u_prev
 *    [11:101]   p_ref (30*3)
 *    [101:104]  p_obs
 *    [104:107]  v_obs
 *    [107]      r_obs
 *    [108:1548] SFC planes: step j, plane m at 108 + (j*M_MAX+m)*4 = [nx,ny,nz,b]
 */

#ifndef NMPC_UAV_AVOIDANCE__NMPC_NODE_HPP_
#define NMPC_UAV_AVOIDANCE__NMPC_NODE_HPP_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <crazyflie_interfaces/msg/full_state.hpp>
#include <crazyflie_interfaces/srv/takeoff.hpp>
#include <crazyflie_interfaces/srv/land.hpp>

#include "nmpc_uav_avoidance/open_solver.hpp"

namespace nmpc_uav {

// ---------------------------------------------------------------------------
//  Constants — must match build_solver.py
// ---------------------------------------------------------------------------
constexpr int    N  = 30;
constexpr int    NX = 8;
constexpr int    NU = 3;
constexpr double Ts = 0.05;

constexpr double GRAVITY   = 9.81;
constexpr double TAU_PHI   = 0.5;
constexpr double TAU_THETA = 0.5;
constexpr double K_PHI     = 1.0;
constexpr double K_THETA   = 1.0;
constexpr double AX = 0.1, AY = 0.1, AZ = 0.2;

// SFC plane budget (per horizon step)
constexpr int M_MAX = 12;

// Parameter layout
constexpr int SFC_START = 108;
constexpr int SFC_BLOCK = 4 * M_MAX * N;          // 1440
constexpr int N_PARAMS  = SFC_START + SFC_BLOCK;  // 1548

constexpr double R_S_MAX = 0.1;   // soft-obstacle ramp, must match build_solver.py

// ---------------------------------------------------------------------------
//  POD types
// ---------------------------------------------------------------------------
struct EulerAngles { double roll, pitch, yaw; };
struct Quaternion  { double x, y, z, w; };

struct SfcPlane {
  std::array<double, 3> n;
  double b;
};

struct SfcPolytope {
  int segment_index{0};
  std::vector<SfcPlane> planes;
};

// ---------------------------------------------------------------------------
//  Free helpers
// ---------------------------------------------------------------------------
EulerAngles quaternion_to_euler(double qx, double qy, double qz, double qw);
Quaternion  euler_to_quaternion(double roll, double pitch, double yaw);
std::array<double, NX> dynamics_step(const std::array<double, NX>& x,
                                     const std::array<double, NU>& u);

// ===========================================================================
//  Node
// ===========================================================================
class NMPCNode : public rclcpp::Node
{
public:
  NMPCNode();
  ~NMPCNode();

private:
  // ---- Callbacks --------------------------------------------------------
  void odom_callback     (const nav_msgs::msg::Odometry::SharedPtr msg);
  void obs_odom_callback (const nav_msgs::msg::Odometry::SharedPtr msg);
  void path_callback     (const nav_msgs::msg::Path::SharedPtr msg);
  void sfc_callback      (const std_msgs::msg::Float64MultiArray::SharedPtr msg);

  // ---- Crazyflie services ----------------------------------------------
  void request_takeoff();
  void request_land();

  // ---- Reference (arc-length, constant-speed virtual target) -----------
  void   rebuild_path_arclength();
  size_t find_segment_index_for_s(double s) const;
  std::array<double, 3> sample_path_at_s(double s) const;
  double find_nearest_path_s(const std::array<double, 3>& p,
                             size_t* nearest_idx) const;
  void   update_reference();

  // ---- SFC helpers ------------------------------------------------------
  bool point_inside_polytope(const std::array<double,3>& p,
                             const std::vector<SfcPlane>& planes,
                             double tol = 1e-3) const;
  int  find_closest_polytope(const std::array<double,3>& p) const;
  int  find_polytope_for_point(const std::array<double,3>& p,
                               int idx_start) const;
  std::vector<int> assign_polytopes_to_horizon();
  void fill_sfc_params(std::vector<double>& params,
                       const std::vector<int>& assignment) const;

  // ---- Control ----------------------------------------------------------
  void control_loop();
  void publish_full_state(const std::array<double, NX>& x_now,
                          const std::vector<double>& u_seq,
                          const std::array<double, 3>& ref_clamp_center);
  void publish_fallback_setpoint();
  void publish_predicted_path(const std::vector<double>& z_star);
  void publish_obstacle_marker();
  void publish_obstacle_diagnostics(const std::vector<double>& z_star);
  void publish_sfc_diagnostics(const std::vector<double>& z_star,
                               const std::vector<int>& assignment);

  // ---- Members ----------------------------------------------------------
  std::unique_ptr<OpEnSolver> solver_;

  // Subscribers
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr     sub_odom_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr     sub_obs_odom_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr         sub_path_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_sfc_;

  // Publishers
  rclcpp::Publisher<crazyflie_interfaces::msg::FullState>::SharedPtr pub_cmd_full_state_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr                  pub_pred_path_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr      pub_obs_marker_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr               pub_solver_time_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr               pub_cost_function_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr     pub_control_input_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr     pub_jerk_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr               pub_obs_viol_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr               pub_obs_dist_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr     pub_pred_obs_viol_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr               pub_sfc_viol_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr     pub_sfc_margin_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr     pub_pred_sfc_viol_;

  // Service clients
  rclcpp::Client<crazyflie_interfaces::srv::Takeoff>::SharedPtr cli_takeoff_;
  rclcpp::Client<crazyflie_interfaces::srv::Land>::SharedPtr    cli_land_;

  rclcpp::TimerBase::SharedPtr timer_;

  // State
  std::array<double, NX> state_  {0,0,0, 0,0,0, 0,0};
  std::array<double, NU> u_prev_ {GRAVITY, 0.0, 0.0};
  double yaw_ = 0.0;
  bool   state_received_ = false;

  // Dynamic obstacle (SOFT; r_obs MUST be inflated)
  std::array<double, 3> obs_pos_ {1.0, 0.0, 1.0};
  std::array<double, 3> obs_vel_ {0.0, 0.0, 0.0};
  double r_obs_ = 0.3;
  bool   obs_received_ = false;

  // SFC (HARD via ALM)
  std::vector<SfcPolytope> sfcs_;
  int  last_poly_idx_ = 0;
  bool sfc_received_  = false;

  // Solver non-convergence handling (Option A: hold on any non-converge)
  int nonconv_streak_     = 0;
  int max_nonconv_streak_ = 1;   // reserved; A holds immediately (streak>=1)

  // Path + arc-length reference tracking
  std::vector<std::array<double, 3>> path_points_;
  std::vector<double> path_s_;
  double path_length_              = 0.0;
  size_t path_idx_                 = 0;
  double last_solve_time_s_        = Ts;
  double ref_progress_s_           = 0.0;
  bool   ref_progress_initialized_ = false;
  bool   use_adaptive_lookahead_   = true;
  double reference_speed_          = 0.10;
  double lookahead_distance_       = 0.12;
  double max_ref_ahead_distance_   = 0.30;

  // Carrot position clamp
  double carrot_max_dev_xy_ = 0.50;
  double carrot_max_dev_z_  = 0.15;

  // Crazyflie takeoff / land state machine
  std::string cf_name_      = "cf231";
  bool   takeoff_requested_ = false;
  bool   takeoff_done_      = false;
  int    settle_counter_    = 0;
  double takeoff_height_    = 0.25;
  double takeoff_duration_  = 3.0;
  double hover_duration_    = 3.0;

  // cmd_full_state carrot lookahead (steps)
  int prediction_lookahead_steps_ = 5;

  // Tunables
  double K_psi_ = 1.0;
};

}  // namespace nmpc_uav

#endif  // NMPC_UAV_AVOIDANCE__NMPC_NODE_HPP_