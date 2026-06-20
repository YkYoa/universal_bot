// ============================================================================
//  OPENARM LOW-LEVEL API IMPLEMENTATION (7 JOINTS)
//  Mã nguồn C++ thực thi đóng gói/gửi ở PC và giải mã/xử lý ở vi điều khiển.
// ============================================================================

#include "arm_api_protocol.h"
#include <iostream>
#include <cstring>
#include <algorithm>

// ----------------------------------------------------------------------------
//  PHẦN 1: CLIENT SIDE (Chạy trên PC / ROS 2 Node)
//  Sử dụng libcurl để gửi dữ liệu nhị phân HTTP POST tới vi điều khiển.
// ----------------------------------------------------------------------------
#ifdef CLIENT_SIDE_REQUIRED
#include <curl/curl.h>

class OpenArmApiClient {
private:
    std::string endpoint_url_;

public:
    explicit OpenArmApiClient(std::string endpoint_url = ARM_API_ENDPOINT)
        : endpoint_url_(std::move(endpoint_url)) {
        // Khởi tạo thư viện curl
        curl_global_init(CURL_GLOBAL_ALL);
    }

    ~OpenArmApiClient() {
        curl_global_cleanup();
    }

    // Hàm tiện ích hỗ trợ đọc dữ liệu trả về từ ESP32
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        size_t total_size = size * nmemb;
        auto* feedback = static_cast<ArmFeedbackPayload*>(userp);
        
        // Nếu ESP32 trả về dữ liệu nhị phân khớp với cấu trúc feedback
        if (total_size == sizeof(ArmFeedbackPayload)) {
            std::memcpy(feedback, contents, sizeof(ArmFeedbackPayload));
        }
        return total_size;
    }

    /**
     * Gửi góc đích điều khiển cho cả 7 khớp của robot
     * @param targets Mảng chứa 7 góc đích (đơn vị: Radians)
     * @param speed Tốc độ giới hạn (1-100%)
     * @param feedback [out] Biến lưu trạng thái phản hồi từ robot
     * @return true nếu gửi thành công và nhận được phản hồi hợp lệ
     */
    bool sendJointsCommand(const float targets[7], int32_t speed, ArmFeedbackPayload& feedback) {
        CURL* curl = curl_easy_init();
        if (!curl) {
            std::cerr << "[PC Client] Lỗi khởi tạo CURL!" << std::endl;
            return false;
        }

        // 1. Tạo payload và nạp dữ liệu
        ArmApiPayload payload;
        payload.command_id = CMD_SET_JOINTS;
        payload.speed_limit = std::max(1, std::min(100, speed));
        std::memcpy(payload.joint_targets, targets, sizeof(payload.joint_targets));
        
        // 2. Tính checksum để bảo vệ dữ liệu truyền dẫn
        payload.checksum = calculate_payload_checksum(&payload);

        // 3. Cấu hình HTTP Request
        curl_easy_setopt(curl, CURLOPT_URL, endpoint_url_.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, (char*)&payload);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, sizeof(payload));

        // Thiết lập header: application/octet-stream (truyền nhị phân trực tiếp)
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        // Thiết lập callback để lưu feedback nhị phân từ ESP32
        std::memset(&feedback, 0, sizeof(feedback));
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &feedback);

        // Thời gian chờ tối đa 1 giây để phản ứng nhanh
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 1000L);

        // 4. Thực thi gửi
        CURLcode res = curl_easy_perform(curl);
        bool success = false;

        if (res != CURLE_OK) {
            std::cerr << "[PC Client] Lỗi kết nối đến robot: " << curl_easy_strerror(res) << std::endl;
        } else {
            long response_code;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
            if (response_code == 200) {
                success = true;
            } else {
                std::cerr << "[PC Client] Robot trả về mã lỗi HTTP: " << response_code << std::endl;
            }
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return success;
    }

    /**
     * Dừng khẩn cấp robot
     */
    bool stopRobot(ArmFeedbackPayload& feedback) {
        float dummy_targets[7] = {0.0f};
        return sendCommandOnly(CMD_STOP, dummy_targets, feedback);
    }

private:
    bool sendCommandOnly(ArmCommandId cmd, const float targets[7], ArmFeedbackPayload& feedback) {
        CURL* curl = curl_easy_init();
        if (!curl) return false;

        ArmApiPayload payload;
        payload.command_id = cmd;
        payload.speed_limit = 0;
        std::memcpy(payload.joint_targets, targets, sizeof(payload.joint_targets));
        payload.checksum = calculate_payload_checksum(&payload);

        curl_easy_setopt(curl, CURLOPT_URL, endpoint_url_.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, (char*)&payload);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, sizeof(payload));

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &feedback);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 1000L);

        CURLcode res = curl_easy_perform(curl);
        bool success = (res == CURLE_OK);

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return success;
    }
};
#endif


// ----------------------------------------------------------------------------
//  PHẦN 2: RECEIVER SIDE (Chạy trên Vi điều khiển / ESP32)
//  Logic minh họa cách giải mã dữ liệu nhị phân nhận từ mạng và điều khiển motor.
// ----------------------------------------------------------------------------

namespace OpenArmReceiver {

    // Định nghĩa giới hạn góc hoạt động an toàn của từng khớp (Soft Limits) dạng Radians
    // Giúp bảo vệ robot không bị bẻ gãy phần cơ khí khi code ROS/RViz2 gửi sai giá trị.
    const float JOINT_MIN_LIMITS[7] = {-1.57f, -1.57f, -2.00f, -1.57f, -1.57f, -3.14f, -3.14f};
    const float JOINT_MAX_LIMITS[7] = { 1.57f,  1.57f,  2.00f,  1.57f,  1.57f,  3.14f,  3.14f};

    // Hàm chuyển góc từ Radians sang góc độ (Degrees) thường dùng cho Servo RC phổ thông
    float radToDeg(float rad) {
        return rad * (180.0f / 3.14159265f);
    }

    // Hàm chuyển góc từ Radians sang độ rộng xung PWM (Microseconds) cho Servo PWM
    // VD: 0 rad -> 1500us, -1.57 rad (-90 deg) -> 500us, +1.57 rad (+90 deg) -> 2500us
    int radToPwmWidth(float rad) {
        float deg = radToDeg(rad);
        // map deg từ [-90, 90] sang [500, 2500] microseconds
        int pulse = 1500 + static_cast<int>((deg / 90.0f) * 1000.0f);
        return std::max(500, std::min(2500, pulse));
    }

    /**
     * Hàm xử lý khi vi điều khiển nhận được gói byte nhị phân thô qua TCP/HTTP
     * @param raw_data Con trỏ trỏ tới vùng đệm dữ liệu vừa nhận được
     * @param len Độ dài của dữ liệu nhận được (bytes)
     */
    void processIncomingBytes(const uint8_t* raw_data, size_t len) {
        // 1. Kiểm tra kích thước gói dữ liệu
        if (len < sizeof(ArmApiPayload)) {
            std::cerr << "[ESP32 Servo] LOI: Kich thuoc payload qua nho!" << std::endl;
            return;
        }

        // 2. Chuyển đổi vùng nhớ thô sang Struct
        ArmApiPayload received_payload;
        std::memcpy(&received_payload, raw_data, sizeof(ArmApiPayload));

        // 3. Kiểm tra mã Checksum
        int32_t expected_checksum = calculate_payload_checksum(&received_payload);
        if (received_payload.checksum != expected_checksum) {
            std::cerr << "[ESP32 Servo] LOI: Sai checksum! Du lieu bi loi." << std::endl;
            return;
        }

        // 4. Xử lý lệnh theo Command ID
        switch (received_payload.command_id) {
            case CMD_STOP:
                std::cout << "[ESP32 Servo] LENH: Dung khan cap toan bo cac khớp!" << std::endl;
                // [Ghi vào Motor]: Thực hiện ngắt xung PWM hoặc hạ torque tại đây.
                break;

            case CMD_SET_JOINTS: {
                std::cout << "[ESP32 Servo] LENH: Cap nhat goc 7 khop. Toc do gioi han: " 
                          << received_payload.speed_limit << "%" << std::endl;

                for (int i = 0; i < 7; ++i) {
                    float target_angle = received_payload.joint_targets[i];

                    // Bảo vệ 1: Giới hạn mềm (Clamp) bảo vệ cơ khí
                    float clamped_angle = std::max(JOINT_MIN_LIMITS[i], 
                                                   std::min(JOINT_MAX_LIMITS[i], target_angle));
                    
                    if (clamped_angle != target_angle) {
                        std::cout << "[ESP32 Servo] Canh bao: Khop " << (i + 1) 
                                  << " vuot qua gioi han! Da clamp tu " << target_angle 
                                  << " ve " << clamped_angle << std::endl;
                    }

                    // Chuyển đổi góc Radians sang xung PWM để ghi xuống phần cứng điều khiển
                    int pwm_pulse = radToPwmWidth(clamped_angle);
                    
                    std::cout << "  - Khop " << (i + 1) << ": " << radToDeg(clamped_angle) 
                              << " do -> Xung PWM: " << pwm_pulse << " us" << std::endl;
                    
                    // [Ghi vào Motor]: Ghi xung pwm_pulse này xuống kênh PWM điều khiển servo (ví dụ: PCA9685 hoặc ESP32 ledcWrite)
                }
                break;
            }

            case CMD_REBOOT:
                std::cout << "[ESP32 Servo] LENH: Khoi dong lai ESP32..." << std::endl;
                // [MCU command]: ESP.restart() trong Arduino
                break;

            default:
                std::cout << "[ESP32 Servo] LOI: Lenh khong xac dinh: " 
                          << received_payload.command_id << std::endl;
                break;
        }
    }
}
