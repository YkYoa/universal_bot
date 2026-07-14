#ifndef HEAD_API_PROTOCOL_H
#define HEAD_API_PROTOCOL_H

#include <stdint.h>

// ============================================================================
//  OPENARM HEAD API PROTOCOL (PAN/TILT, MG996R + CUSTOM ANALOG TAP)
//  Giao thuc nhi phan giua PC (head_motor_driver_node) va Vi dieu khien (ESP32).
//  Khac voi arm_api: servo khong co feedback goc chuan, phai doc ADC tho tu
//  day potentiometer noi bo da hàn tay ra ngoai. Moi quy doi ADC -> radian
//  thuc hien o phia PC (xem calibration.hpp), ESP32 chi gui/nhan gia tri tho.
// ============================================================================

#define HEAD_API_ENDPOINT "http://192.168.1.101:80/api/head/command"

// ----------------------------------------------------------------------------
// 1. Command IDs
// ----------------------------------------------------------------------------
typedef enum {
    HEAD_CMD_STOP        = 0,  // Ngung xung PWM ca 2 khop (giu nguyen vi tri hien tai)
    HEAD_CMD_SET_POSITION = 1, // Gui pulse-width dich cho pan + tilt
    HEAD_CMD_REBOOT      = 9   // Khoi dong lai ESP32
} HeadCommandId;

// ----------------------------------------------------------------------------
// 2. PC -> MCU Payload
//    ESP32 chi biet pulse-width (us), khong biet radian/calibration.
//    Quy doi rad -> pulse-width da lam xong o PC truoc khi gui xuong.
// ----------------------------------------------------------------------------
typedef struct {
    int32_t  command_id;      // HeadCommandId
    uint16_t pan_pulse_us;    // Do rong xung PWM cho pan, thuong 500-2500us
    uint16_t tilt_pulse_us;   // Do rong xung PWM cho tilt
    int32_t  checksum;        // XOR cac truong phia truoc
} HeadApiPayload;

// ----------------------------------------------------------------------------
// 3. MCU -> PC Feedback
//    adc_raw la gia tri tho 12-bit (0-4095) doc tu day wiper cua potentiometer
//    da tap. PC tu quy doi sang radian bang calibration rieng cho tung khop/con
//    servo (xem head_calibration.yaml) - KHONG quy doi o firmware.
// ----------------------------------------------------------------------------
typedef struct {
    int32_t  status;          // 0 = OK, khac 0 = ma loi phan cung (vd: mat ket noi ADC)
    uint16_t pan_adc_raw;     // ADC tho, chua quy doi
    uint16_t tilt_adc_raw;    // ADC tho, chua quy doi
} HeadFeedbackPayload;

// ----------------------------------------------------------------------------
// 4. Checksum helper (giong pattern arm_api_protocol.h)
// ----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

static inline int32_t calculate_head_payload_checksum(const HeadApiPayload* payload) {
    int32_t sum = 0;
    sum ^= payload->command_id;
    sum ^= payload->pan_pulse_us;
    sum ^= payload->tilt_pulse_us;
    return sum;
}

#ifdef __cplusplus
}
#endif

#endif // HEAD_API_PROTOCOL_H
