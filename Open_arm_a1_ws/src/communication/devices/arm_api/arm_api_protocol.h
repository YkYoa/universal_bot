#ifndef ARM_API_PROTOCOL_H
#define ARM_API_PROTOCOL_H

#include <stdint.h>

// ============================================================================
//  OPENARM LOW-LEVEL API PROTOCOL (7 JOINTS)
//  Giao thức truyền dữ liệu nhị phân giữa PC (ROS 2) và Vi điều khiển (ESP32)
//  Đã loại bỏ bộ kẹp (gripper), tập trung điều khiển 7 khớp chính.
// ============================================================================

// Địa chỉ IP và Port mặc định của vi điều khiển lắng nghe API HTTP
#define ARM_API_ENDPOINT "http://192.168.1.100:80/api/arm/command"

// ----------------------------------------------------------------------------
// 1. Unified Command IDs (Mã lệnh điều khiển)
// ----------------------------------------------------------------------------
typedef enum {
    CMD_STOP            = 0,   // Dừng khẩn cấp toàn bộ các khớp (giữ vị trí hoặc xả torque)
    CMD_SET_JOINTS      = 1,   // Gửi góc đích đồng thời cho cả 7 khớp
    CMD_SET_SINGLE_JOINT = 2,  // Chỉ điều khiển một khớp duy nhất (truyền khớp qua extra_param)
    CMD_CALIBRATE       = 3,   // Đặt vị trí hiện tại làm mốc 0 (home calibration)
    CMD_REBOOT          = 9    // Khởi động lại vi điều khiển
} ArmCommandId;

// ----------------------------------------------------------------------------
// 2. PC to MCU Payload Layout (Dữ liệu gửi từ PC xuống Vi điều khiển)
//    Kích thước cố định: 40 bytes (để đảm bảo không bị alignment issue)
// ----------------------------------------------------------------------------
typedef struct {
    int32_t command_id;         // Giá trị từ enum ArmCommandId
    int32_t speed_limit;        // Giới hạn tốc độ lớn nhất (%) từ 1 đến 100
    float joint_targets[7];     // Góc đích cho 7 khớp (đơn vị: Radians / radian)
    int32_t checksum;           // Mã kiểm tra lỗi dữ liệu (XOR toàn bộ các trường trước)
} ArmApiPayload;

// ----------------------------------------------------------------------------
// 3. MCU to PC Feedback Layout (Dữ liệu phản hồi ngược lại PC)
//    Kích thước cố định: 36 bytes
// ----------------------------------------------------------------------------
typedef struct {
    int32_t status;             // Trạng thái hệ thống: 0 = OK, còn lại là mã lỗi
    float joint_states[7];      // Góc hiện tại của 7 khớp đọc về từ Encoder (Radians)
    int32_t is_healthy;         // Trạng thái sức khỏe: 1 = Khỏe mạnh, 0 = Lỗi quá nhiệt/quá tải
} ArmFeedbackPayload;

// ----------------------------------------------------------------------------
// 4. Tiện ích Helper để tính toán Checksum đơn giản
// ----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

static inline int32_t calculate_payload_checksum(const ArmApiPayload* payload) {
    const int32_t* raw_ptr = (const int32_t*)payload;
    int32_t sum = 0;
    // XOR tất cả các phần tử ngoại trừ trường checksum cuối cùng (9 phần tử đầu)
    for (int i = 0; i < 9; i++) {
        sum ^= raw_ptr[i];
    }
    return sum;
}

#ifdef __cplusplus
}
#endif

#endif // ARM_API_PROTOCOL_H
