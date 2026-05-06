#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <atomic>

using namespace std::chrono_literals;

class CameraTopicPub : public rclcpp::Node
{
public:
    CameraTopicPub() : Node("camera_topic_pub"), 
                       count_left_rgb_(0), count_right_rgb_(0), count_head_rgb_(0),
                       count_left_depth_(0), count_right_depth_(0), count_head_depth_(0)
    {
        // ########################### 1. 创建6个独立回调组（6线程核心：1组1线程）###########################
        // 彩色图回调组（原有3个）
        group_left_rgb_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        group_right_rgb_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        group_head_rgb_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        // 深度图回调组（新增3个，与彩色图完全解耦）
        group_left_depth_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        group_right_depth_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        group_head_depth_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

        // ########################### 2. 创建6个发布者（3彩色+3深度）###########################
        // 彩色图发布者（原有）
        pub_left_rgb_ = this->create_publisher<sensor_msgs::msg::Image>("/camera1/camera_left/color/image_rect_raw", 10);
        pub_right_rgb_ = this->create_publisher<sensor_msgs::msg::Image>("/camera2/camera_right/color/image_rect_raw", 10);
        pub_head_rgb_ = this->create_publisher<sensor_msgs::msg::Image>("/camera3/camera_head/color/image_raw", 10);
        // 深度图发布者（新增，严格按指定Topic名称，16UC1对应mono16编码）
        pub_left_depth_ = this->create_publisher<sensor_msgs::msg::Image>("/camera1/camera_left/depth/image_rect_raw", 10);
        pub_right_depth_ = this->create_publisher<sensor_msgs::msg::Image>("/camera2/camera_right/depth/image_rect_raw", 10);
        pub_head_depth_ = this->create_publisher<sensor_msgs::msg::Image>("/camera3/camera_head/depth/image_rect_raw", 10);

        // ########################### 3. 创建6个独立定时器（3彩色+3深度，各绑定专属回调组）###########################
        // 彩色图定时器（原有，30fps=33.33ms）
        timer_left_rgb_ = this->create_wall_timer(33.33ms, std::bind(&CameraTopicPub::publish_left_rgb, this), group_left_rgb_);
        timer_right_rgb_ = this->create_wall_timer(33.33ms, std::bind(&CameraTopicPub::publish_right_rgb, this), group_right_rgb_);
        timer_head_rgb_ = this->create_wall_timer(33.33ms, std::bind(&CameraTopicPub::publish_head_rgb, this), group_head_rgb_);
        // 深度图定时器（新增，同频率，绑定深度专属回调组）
        timer_left_depth_ = this->create_wall_timer(33.33ms, std::bind(&CameraTopicPub::publish_left_depth, this), group_left_depth_);
        timer_right_depth_ = this->create_wall_timer(33.33ms, std::bind(&CameraTopicPub::publish_right_depth, this), group_right_depth_);
        timer_head_depth_ = this->create_wall_timer(33.33ms, std::bind(&CameraTopicPub::publish_head_depth, this), group_head_depth_);
        
        RCLCPP_INFO(this->get_logger(), "多线程相机节点已启动，6路Topic(3RGB+3Depth)，每路独立线程，频率30fps");
    }

private:
    // ########################### 彩色图发布回调（原有，无修改）###########################
    void publish_left_rgb() {
        auto now = this->get_clock()->now();
        pub_left_rgb_->publish(*generate_rgb_image_msg(now, count_left_rgb_++, "Left"));
    }
    void publish_right_rgb() {
        auto now = this->get_clock()->now();
        pub_right_rgb_->publish(*generate_rgb_image_msg(now, count_right_rgb_++, "Right"));
    }
    void publish_head_rgb() {
        auto now = this->get_clock()->now();
        pub_head_rgb_->publish(*generate_rgb_image_msg(now, count_head_rgb_++, "Head"));
    }

    // ########################### 深度图发布回调（新增，与彩色图逻辑一致）###########################
    void publish_left_depth() {
        auto now = this->get_clock()->now();
        pub_left_depth_->publish(*generate_depth_image_msg(now, count_left_depth_++, "Left"));
    }
    void publish_right_depth() {
        auto now = this->get_clock()->now();
        pub_right_depth_->publish(*generate_depth_image_msg(now, count_right_depth_++, "Right"));
    }
    void publish_head_depth() {
        auto now = this->get_clock()->now();
        pub_head_depth_->publish(*generate_depth_image_msg(now, count_head_depth_++, "Head"));
    }

    // ########################### 生成彩色图消息（原有，无修改，BGR8格式）###########################
    sensor_msgs::msg::Image::SharedPtr generate_rgb_image_msg(const rclcpp::Time& stamp, int count, const std::string& label)
    {
        cv::Mat image(480, 640, CV_8UC3, cv::Scalar(255, 255, 255));
        std::string text = std::to_string(count);
        
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 4.0;
        int thickness = 10;
        int baseline = 0;

        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
        cv::Point textOrg((image.cols - textSize.width) / 2, (image.rows + textSize.height) / 2);
        cv::putText(image, text, textOrg, fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);

        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image).toImageMsg();
        msg->header.stamp = stamp;
        msg->header.frame_id = label + "_rgb_frame";
        return msg;
    }

    // ########################### 生成深度图消息（新增，核心：16UC1格式，对应ROS mono16）###########################
    sensor_msgs::msg::Image::SharedPtr generate_depth_image_msg(const rclcpp::Time& stamp, int count, const std::string& label)
    {
        // 创建16UC1深度图（640x480，与彩色图同分辨率），初始深度值1000（单位：mm，常规深度图取值）
        cv::Mat depth_image(480, 640, CV_16UC1, cv::Scalar(1000));
        std::string text = std::to_string(count); // 绘制计数，区分帧序列

        // 16UC1图绘制文本：需先转换为8UC3临时图（OpenCV对16位单通道图绘制支持有限）
        cv::Mat temp_8uc3;
        depth_image.convertTo(temp_8uc3, CV_8UC3, 1.0/256); // 16位转8位，保证亮度正常
        // 绘制文本（黑色，参数与彩色图一致）
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 4.0;
        int thickness = 10;
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
        cv::Point textOrg((depth_image.cols - textSize.width) / 2, (depth_image.rows + textSize.height) / 2);
        cv::putText(temp_8uc3, text, textOrg, fontFace, fontScale, cv::Scalar(0, 0, 0), thickness);
        // 转回16UC1深度图
        temp_8uc3.convertTo(depth_image, CV_16UC1, 256);

        // 转换为ROS消息：关键指定编码为mono16（对应OpenCV 16UC1，ROS深度图标准编码）
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono16", depth_image).toImageMsg();
        msg->header.stamp = stamp; // 与对应彩色图同时间戳，保证时间同步
        msg->header.frame_id = label + "_depth_frame"; // 独立坐标系，区分彩色图
        return msg;
    }

    // ########################### 成员变量：6发布者+6定时器+6回调组+6原子计数（全解耦）###########################
    // 彩色图 - 发布者（原有）
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_left_rgb_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_right_rgb_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_head_rgb_;
    // 深度图 - 发布者（新增）
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_left_depth_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_right_depth_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_head_depth_;

    // 彩色图 - 定时器（原有）
    rclcpp::TimerBase::SharedPtr timer_left_rgb_;
    rclcpp::TimerBase::SharedPtr timer_right_rgb_;
    rclcpp::TimerBase::SharedPtr timer_head_rgb_;
    // 深度图 - 定时器（新增）
    rclcpp::TimerBase::SharedPtr timer_left_depth_;
    rclcpp::TimerBase::SharedPtr timer_right_depth_;
    rclcpp::TimerBase::SharedPtr timer_head_depth_;

    // 彩色图 - 回调组（原有，Reentrant类型：支持重入，独立线程执行）
    rclcpp::CallbackGroup::SharedPtr group_left_rgb_;
    rclcpp::CallbackGroup::SharedPtr group_right_rgb_;
    rclcpp::CallbackGroup::SharedPtr group_head_rgb_;
    // 深度图 - 回调组（新增，同类型，与彩色图完全独立）
    rclcpp::CallbackGroup::SharedPtr group_left_depth_;
    rclcpp::CallbackGroup::SharedPtr group_right_depth_;
    rclcpp::CallbackGroup::SharedPtr group_head_depth_;
    
    // 原子计数（原有3个+新增3个，多线程安全，避免计数混乱）
    std::atomic<int> count_left_rgb_;
    std::atomic<int> count_right_rgb_;
    std::atomic<int> count_head_rgb_;
    std::atomic<int> count_left_depth_;
    std::atomic<int> count_right_depth_;
    std::atomic<int> count_head_depth_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraTopicPub>();
    // 多线程执行器（核心：自动为6个Reentrant回调组分配6个独立线程，无需手动创建线程）
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin(); // 启动事件循环，6线程并发执行
    rclcpp::shutdown();
    return 0;
}
