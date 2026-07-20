// linear_dynamic_obstacle_node.cpp
//
// 두 점 A <-> B 사이를 등속 왕복하는 "선형 동적 장애물" 1개.
// NMPC solver는 그대로 두고, /obstacle/odom 에 현재 위치 + 현재 속도를 발행한다.
// (solver는 obs_pos + j*Ts*obs_vel 로 미래 위치를 예측하므로, 속도를
//  정확히 실어 보내야 예측 회피가 동작한다.)
//
// Publishes:  /obstacle/odom    (nav_msgs/Odometry)            — solver 입력
//             /obstacle/markers (visualization_msgs/MarkerArray) — 시각화
//
// 파라미터:
//   speed         [m/s]  왕복 속도 (기본 0.15)
//   r_obs         [m]    안전 반경 (마커용; solver의 r_obs와 맞출 것)
//   ax,ay,az / bx,by,bz  두 끝점 A, B

#include <chrono>
#include <memory>
#include <cmath>
#include <array>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

using namespace std::chrono_literals;

class LinearDynamicObstacleNode : public rclcpp::Node
{
public:
    LinearDynamicObstacleNode()
    : Node("linear_dynamic_obstacle_node")
    {
        // ---- 파라미터 ----
        this->declare_parameter<double>("speed", 0.15);   // 왕복 속도 [m/s]
        this->declare_parameter<double>("r_obs", 0.15);   // 안전 반경 (마커용)
        this->declare_parameter<double>("ax", 0.31);
        this->declare_parameter<double>("ay", 0.06);
        this->declare_parameter<double>("az", 0.15);
        this->declare_parameter<double>("bx", 1.49);
        this->declare_parameter<double>("by", 0.06);
        this->declare_parameter<double>("bz", 0.15);

        speed_ = this->get_parameter("speed").as_double();
        r_obs_ = this->get_parameter("r_obs").as_double();
        A_ = { this->get_parameter("ax").as_double(),
               this->get_parameter("ay").as_double(),
               this->get_parameter("az").as_double() };
        B_ = { this->get_parameter("bx").as_double(),
               this->get_parameter("by").as_double(),
               this->get_parameter("bz").as_double() };

        // 진행 방향 단위벡터와 전체 길이
        const double dx = B_[0]-A_[0], dy = B_[1]-A_[1], dz = B_[2]-A_[2];
        seg_len_ = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (seg_len_ > 1e-9) {
            dir_ = { dx/seg_len_, dy/seg_len_, dz/seg_len_ };
        } else {
            dir_ = {0.0, 0.0, 0.0};   // A==B 방어
        }

        pos_ = A_;          // A에서 시작
        s_   = 0.0;         // A로부터의 호 길이
        going_to_B_ = true; // 처음엔 B 방향

        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "/obstacle/odom", 10);
        markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/obstacle/markers", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(dt_ * 1000)),
            std::bind(&LinearDynamicObstacleNode::tick, this));

        RCLCPP_INFO(this->get_logger(),
            "Linear dynamic obstacle: A=[%.2f,%.2f,%.2f] <-> B=[%.2f,%.2f,%.2f], "
            "speed=%.3f m/s, len=%.3f m, r=%.2f",
            A_[0],A_[1],A_[2], B_[0],B_[1],B_[2], speed_, seg_len_, r_obs_);
    }

private:
    void tick()
    {
        // ---- 위치/속도 적분 (등속 왕복) ----
        const double ds = speed_ * dt_;
        if (going_to_B_) {
            s_ += ds;
            if (s_ >= seg_len_) { s_ = seg_len_; going_to_B_ = false; }
        } else {
            s_ -= ds;
            if (s_ <= 0.0)      { s_ = 0.0;      going_to_B_ = true; }
        }

        // 현재 위치 = A + s * dir
        pos_ = { A_[0] + dir_[0]*s_,
                 A_[1] + dir_[1]*s_,
                 A_[2] + dir_[2]*s_ };

        // 현재 속도 벡터 (방향에 따라 +dir 또는 -dir)
        const double sgn = going_to_B_ ? +1.0 : -1.0;
        vel_ = { dir_[0]*speed_*sgn, dir_[1]*speed_*sgn, dir_[2]*speed_*sgn };

        publish_odom();
        publish_markers();
    }

    void publish_odom()
    {
        nav_msgs::msg::Odometry msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = "world";
        msg.child_frame_id = "obstacle";

        msg.pose.pose.position.x = pos_[0];
        msg.pose.pose.position.y = pos_[1];
        msg.pose.pose.position.z = pos_[2];
        msg.pose.pose.orientation.w = 1.0;

        // 동적: 현재 속도 벡터를 실어 보냄 (NMPC 예측 회피에 필수)
        msg.twist.twist.linear.x = vel_[0];
        msg.twist.twist.linear.y = vel_[1];
        msg.twist.twist.linear.z = vel_[2];
        msg.twist.twist.angular.x = 0.0;
        msg.twist.twist.angular.y = 0.0;
        msg.twist.twist.angular.z = 0.0;

        odom_pub_->publish(msg);
    }

    void publish_markers()
    {
        visualization_msgs::msg::MarkerArray arr;

        // 움직이는 장애물 구
        {
            visualization_msgs::msg::Marker m;
            m.header.stamp = this->now();
            m.header.frame_id = "world";
            m.ns = "dynamic_obstacle";
            m.id = 0;
            m.type = visualization_msgs::msg::Marker::SPHERE;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x = pos_[0];
            m.pose.position.y = pos_[1];
            m.pose.position.z = pos_[2];
            m.pose.orientation.w = 1.0;
            const double d = 2.0 * r_obs_;
            m.scale.x = d; m.scale.y = d; m.scale.z = d;
            m.color.r = 1.0f; m.color.g = 0.0f; m.color.b = 0.0f; m.color.a = 0.75f;
            m.lifetime = rclcpp::Duration(0, 0);
            arr.markers.push_back(m);
        }

        // 왕복 경로 라인 (A-B)
        {
            visualization_msgs::msg::Marker line;
            line.header.stamp = this->now();
            line.header.frame_id = "world";
            line.ns = "obstacle_path";
            line.id = 1;
            line.type = visualization_msgs::msg::Marker::LINE_STRIP;
            line.action = visualization_msgs::msg::Marker::ADD;
            line.pose.orientation.w = 1.0;
            line.scale.x = 0.01;  // 선 두께
            line.color.r = 0.2f; line.color.g = 0.6f; line.color.b = 1.0f; line.color.a = 0.8f;
            geometry_msgs::msg::Point pa, pb;
            pa.x = A_[0]; pa.y = A_[1]; pa.z = A_[2];
            pb.x = B_[0]; pb.y = B_[1]; pb.z = B_[2];
            line.points.push_back(pa);
            line.points.push_back(pb);
            line.lifetime = rclcpp::Duration(0, 0);
            arr.markers.push_back(line);
        }

        markers_pub_->publish(arr);
    }

    // 파라미터
    double speed_ = 0.15;
    double r_obs_ = 0.15;
    std::array<double,3> A_{}, B_{};

    // 운동 상태
    std::array<double,3> dir_{};
    std::array<double,3> pos_{}, vel_{};
    double seg_len_ = 0.0;
    double s_ = 0.0;
    bool going_to_B_ = true;

    static constexpr double dt_ = 0.05;  // 20 Hz

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LinearDynamicObstacleNode>());
    rclcpp::shutdown();
    return 0;
}