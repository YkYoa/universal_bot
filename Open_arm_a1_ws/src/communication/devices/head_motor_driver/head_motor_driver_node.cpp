#include "head_motor_driver_node.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <thread>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>

namespace {

constexpr int kLoopHz = 100;
constexpr double kLoopPeriodS = 1.0 / kLoopHz;
constexpr double kWatchdogS = 0.25;

double nowSeconds()
{
  return std::chrono::duration<double>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

/* Minimal flat-JSON field extraction, mirroring openarm_hardware's
 * HeadHW parser - both ends of this socket are written together for one
 * fixed schema, so a full JSON library is unnecessary overhead here. */

std::optional<double> extractNumber(const std::string& line, const std::string& key)
{
  std::string needle = "\"" + key + "\":";
  auto pos = line.find(needle);
  if (pos == std::string::npos) return std::nullopt;
  pos += needle.size();
  auto end = line.find_first_of(",}", pos);
  if (end == std::string::npos) return std::nullopt;
  try {
    return std::stod(line.substr(pos, end - pos));
  } catch (...) {
    return std::nullopt;
  }
}

JointCalibration loadJointCalibration(const YAML::Node& node)
{
  JointCalibration cal;
  cal.min_angle_rad = node["min_angle_rad"].as<double>();
  cal.max_angle_rad = node["max_angle_rad"].as<double>();
  cal.pwm_min_us = node["pwm_min_us"].as<uint16_t>();
  cal.pwm_max_us = node["pwm_max_us"].as<uint16_t>();
  cal.invert = node["invert"].as<bool>(false);
  cal.center_deg = node["center_deg"].as<double>();
  cal.pwm_to_deg_slope = node["pwm_to_deg_slope"].as<double>();
  cal.pwm_to_deg_intercept = node["pwm_to_deg_intercept"].as<double>();
  cal.adc_to_deg_slope = node["adc_to_deg_slope"].as<double>();
  cal.adc_to_deg_intercept = node["adc_to_deg_intercept"].as<double>();
  cal.max_velocity_rad_s = node["max_velocity_rad_s"].as<double>(cal.max_velocity_rad_s);
  cal.max_accel_rad_s2 = node["max_accel_rad_s2"].as<double>(cal.max_accel_rad_s2);

  if (const YAML::Node& pts = node["points"]) {
    for (const auto& pt : pts) {
      cal.points.push_back({pt["adc"].as<double>(), pt["deg"].as<double>()});
    }
    std::sort(cal.points.begin(), cal.points.end(),
              [](const CalibrationPoint& a, const CalibrationPoint& b) { return a.adc < b.adc; });
  }

  return cal;
}

// Linear interpolation through a sorted (ascending by adc) lookup table,
// clamped at the ends. Mirrors auto_calibrate.py's lut_lookup() exactly -
// keep both in sync if this changes.
double lutLookupDeg(const std::vector<CalibrationPoint>& points, double adc)
{
  if (adc <= points.front().adc) return points.front().deg;
  if (adc >= points.back().adc) return points.back().deg;
  for (std::size_t i = 0; i + 1 < points.size(); ++i) {
    const auto& p0 = points[i];
    const auto& p1 = points[i + 1];
    if (adc >= p0.adc && adc <= p1.adc) {
      if (p1.adc == p0.adc) return p0.deg;
      double t = (adc - p0.adc) / (p1.adc - p0.adc);
      return p0.deg + t * (p1.deg - p0.deg);
    }
  }
  return points.back().deg;
}

}  // namespace

uint16_t JointCalibration::rad_to_pwm(double rad) const
{
  rad = std::max(min_angle_rad, std::min(max_angle_rad, rad));
  double signed_rad = invert ? -rad : rad;
  double deg = center_deg + signed_rad * 180.0 / M_PI;
  double pwm = (deg - pwm_to_deg_intercept) / pwm_to_deg_slope;
  pwm = std::max<double>(pwm_min_us, std::min<double>(pwm_max_us, pwm));
  return static_cast<uint16_t>(std::lround(pwm));
}

double JointCalibration::adc_to_rad(int adc) const
{
  // Multi-point lookup table (from auto_calibrate.py) corrects
  // potentiometer nonlinearity the 2-point linear fit can't - falls back
  // to the linear fit for a joint not yet calibrated with the new tool.
  double deg = points.empty()
    ? adc_to_deg_slope * adc + adc_to_deg_intercept
    : lutLookupDeg(points, static_cast<double>(adc));
  double rad = (deg - center_deg) * M_PI / 180.0;
  return invert ? -rad : rad;
}

double EmaFilter::update(double sample)
{
  if (!value_.has_value()) {
    value_ = sample;
  } else {
    value_ = alpha_ * sample + (1.0 - alpha_) * (*value_);
  }
  return *value_;
}

// ---------------------------------------------------------------------------
// Stm32Link
// ---------------------------------------------------------------------------

Stm32Link::Stm32Link(std::string host, uint16_t port)
: host_(std::move(host)), port_(port)
{
}

Stm32Link::~Stm32Link()
{
  closeSocket();
}

void Stm32Link::closeSocket()
{
  if (sockfd_ >= 0) {
    ::close(sockfd_);
    sockfd_ = -1;
  }
}

bool Stm32Link::ensureConnected()
{
  if (sockfd_ >= 0) return true;

  struct addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo* res = nullptr;
  if (getaddrinfo(host_.c_str(), std::to_string(port_).c_str(), &hints, &res) != 0) {
    return false;
  }

  int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  if (fd < 0) {
    freeaddrinfo(res);
    return false;
  }

  struct timeval connect_tv{1, 0};
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &connect_tv, sizeof(connect_tv));

  if (::connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
    freeaddrinfo(res);
    ::close(fd);
    return false;
  }
  freeaddrinfo(res);

  struct timeval io_tv{0, 300000};  // 300ms
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &io_tv, sizeof(io_tv));
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &io_tv, sizeof(io_tv));

  sockfd_ = fd;
  return true;
}

std::optional<std::string> Stm32Link::command(const std::string& line)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (!ensureConnected()) {
    return std::nullopt;
  }

  std::string out = line + "\r\n";
  if (::send(sockfd_, out.c_str(), out.size(), MSG_NOSIGNAL) < 0) {
    closeSocket();
    return std::nullopt;
  }

  std::string buf;
  char chunk[256];
  double deadline = nowSeconds() + 0.3;
  while (nowSeconds() < deadline) {
    ssize_t n = ::recv(sockfd_, chunk, sizeof(chunk), 0);
    if (n > 0) {
      buf.append(chunk, static_cast<size_t>(n));
      if (buf.find('\n') != std::string::npos) break;
    } else if (n == 0) {
      closeSocket();
      return std::nullopt;
    } else {
      if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
      closeSocket();
      return std::nullopt;
    }
  }
  if (buf.empty()) {
    closeSocket();
    return std::nullopt;
  }

  while (!buf.empty() && (buf.back() == '\n' || buf.back() == '\r')) buf.pop_back();
  return buf;
}

// ---------------------------------------------------------------------------
// HeadMotorDriver
// ---------------------------------------------------------------------------

HeadMotorDriver::HeadMotorDriver(std::string calibration_path, std::string stm32_host,
                                   uint16_t stm32_port, std::string uds_path)
: pan_filter_(0.3), tilt_filter_(0.3), link_(std::move(stm32_host), stm32_port),
  uds_path_(std::move(uds_path))
{
  YAML::Node cfg = YAML::LoadFile(calibration_path);
  pan_cal_ = loadJointCalibration(cfg["pan"]);
  tilt_cal_ = loadJointCalibration(cfg["tilt"]);
  stall_error_threshold_rad_ = cfg["stall_error_threshold_rad"].as<double>();
  stall_timeout_s_ = cfg["stall_timeout_ms"].as<double>() / 1000.0;
  double alpha = cfg["position_filter_alpha"].as<double>();
  pan_filter_ = EmaFilter(alpha);
  tilt_filter_ = EmaFilter(alpha);

  pan_profile_ = TrapezoidalProfile(pan_cal_.max_velocity_rad_s, pan_cal_.max_accel_rad_s2);
  tilt_profile_ = TrapezoidalProfile(tilt_cal_.max_velocity_rad_s, tilt_cal_.max_accel_rad_s2);
}

void HeadMotorDriver::udsServerLoop()
{
  ::unlink(uds_path_.c_str());

  listen_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
  struct sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, uds_path_.c_str(), sizeof(addr.sun_path) - 1);

  bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr));
  listen(listen_fd_, 1);
  std::cout << "[head_motor_driver] Socket bound at " << uds_path_ << std::endl;

  while (running_) {
    int conn_fd = accept(listen_fd_, nullptr, nullptr);
    if (conn_fd < 0) {
      if (!running_) break;
      continue;
    }
    std::cout << "[head_motor_driver] Hardware interface connected" << std::endl;

    {
      std::lock_guard<std::mutex> lock(uds_client_mutex_);
      uds_client_fd_ = conn_fd;
    }
    readCmdsFrom(conn_fd);
    {
      std::lock_guard<std::mutex> lock(uds_client_mutex_);
      if (uds_client_fd_ == conn_fd) uds_client_fd_ = -1;
    }
    ::close(conn_fd);
    std::cout << "[head_motor_driver] Hardware interface disconnected, awaiting reconnect"
              << std::endl;
  }
}

void HeadMotorDriver::readCmdsFrom(int conn_fd)
{
  struct timeval tv{1, 0};
  setsockopt(conn_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  std::string buf;
  char chunk[4096];
  while (running_) {
    ssize_t n = ::recv(conn_fd, chunk, sizeof(chunk), 0);
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
      break;
    }
    if (n == 0) break;
    buf.append(chunk, static_cast<size_t>(n));

    auto newline = buf.find('\n');
    while (newline != std::string::npos) {
      std::string line = buf.substr(0, newline);
      buf.erase(0, newline + 1);
      if (line.find("\"type\":\"cmd\"") != std::string::npos) {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        if (auto v = extractNumber(line, "pan_cmd")) pan_cmd_ = *v;
        if (auto v = extractNumber(line, "tilt_cmd")) tilt_cmd_ = *v;
        last_cmd_time_s_ = nowSeconds();
      }
      newline = buf.find('\n');
    }
  }
}

void HeadMotorDriver::sendState(double pan_rad, double tilt_rad, bool is_healthy,
                                  const std::string& error_msg)
{
  ++seq_;
  std::ostringstream oss;
  oss << "{\"type\":\"state\",\"seq\":" << seq_
      << ",\"timestamp\":" << nowSeconds()
      << ",\"pan_pos\":" << pan_rad
      << ",\"tilt_pos\":" << tilt_rad
      << ",\"pan_vel\":0.0,\"tilt_vel\":0.0,\"pan_load\":0.0,\"tilt_load\":0.0"
      << ",\"is_healthy\":" << (is_healthy ? "true" : "false")
      << ",\"error_code\":" << (is_healthy ? 0 : 1)
      << ",\"error_msg\":\"" << error_msg << "\"}\n";
  std::string line = oss.str();

  std::lock_guard<std::mutex> lock(uds_client_mutex_);
  if (uds_client_fd_ < 0) return;
  if (::send(uds_client_fd_, line.c_str(), line.size(), MSG_NOSIGNAL) < 0) {
    uds_client_fd_ = -1;
  }
}

std::optional<int> HeadMotorDriver::parseAdc(const std::optional<std::string>& reply)
{
  if (!reply.has_value()) return std::nullopt;
  auto pos = reply->find("ADC=");
  if (pos == std::string::npos) return std::nullopt;
  pos += 4;
  auto end = reply->find_first_not_of("-0123456789", pos);
  try {
    return std::stoi(reply->substr(pos, end == std::string::npos ? std::string::npos : end - pos));
  } catch (...) {
    return std::nullopt;
  }
}

void HeadMotorDriver::controlLoop()
{
  while (running_) {
    double loop_start = nowSeconds();

    double pan_cmd, tilt_cmd;
    std::optional<double> since_cmd;
    {
      std::lock_guard<std::mutex> lock(cmd_mutex_);
      pan_cmd = pan_cmd_;
      tilt_cmd = tilt_cmd_;
      if (last_cmd_time_s_ > 0.0) since_cmd = nowSeconds() - last_cmd_time_s_;
    }
    bool watchdog_tripped = since_cmd.has_value() && *since_cmd > kWatchdogS;

    // Rate-limit the raw commanded angle before converting to PWM, so a
    // step change (single-point RViz goal, coarse trajectory) becomes
    // smooth accel/cruise/decel motion instead of an instant PWM jump.
    if (!profile_initialized_) {
      pan_profile_.reset(pan_cmd);
      tilt_profile_.reset(tilt_cmd);
      last_profile_tick_s_ = loop_start;
      profile_initialized_ = true;
    }
    double profile_dt = loop_start - last_profile_tick_s_;
    last_profile_tick_s_ = loop_start;
    double pan_profiled = pan_profile_.step(pan_cmd, profile_dt);
    double tilt_profiled = tilt_profile_.step(tilt_cmd, profile_dt);

    uint16_t pan_pwm = pan_cal_.rad_to_pwm(pan_profiled);
    uint16_t tilt_pwm = tilt_cal_.rad_to_pwm(tilt_profiled);

    auto reply1 = link_.command("SERVO 0 " + std::to_string(pan_pwm));
    auto reply0 = link_.command("SERVO 1 " + std::to_string(tilt_pwm));

    bool connected = reply0.has_value() && reply1.has_value();
    auto pan_adc = parseAdc(reply0);
    auto tilt_adc = parseAdc(reply1);

    if (connected && pan_adc.has_value() && tilt_adc.has_value()) {
      double pan_rad = pan_filter_.update(pan_cal_.adc_to_rad(*pan_adc));
      double tilt_rad = tilt_filter_.update(tilt_cal_.adc_to_rad(*tilt_adc));

      bool stalled = std::abs(pan_profiled - pan_rad) > stall_error_threshold_rad_ ||
                     std::abs(tilt_profiled - tilt_rad) > stall_error_threshold_rad_;
      if (stalled) {
        if (!stall_since_s_.has_value()) stall_since_s_ = nowSeconds();
      } else if (!watchdog_tripped) {
        stall_since_s_.reset();
      }

      bool stall_timeout_hit = stall_since_s_.has_value() &&
        (nowSeconds() - *stall_since_s_ > stall_timeout_s_);

      sendState(pan_rad, tilt_rad, !stall_timeout_hit,
                stall_timeout_hit ? "stall detected" : "");
    } else {
      sendState(0.0, 0.0, false, "STM32 unreachable");
    }

    double elapsed = nowSeconds() - loop_start;
    double remaining = kLoopPeriodS - elapsed;
    if (remaining > 0.0) {
      std::this_thread::sleep_for(std::chrono::duration<double>(remaining));
    }
  }
}

void HeadMotorDriver::shutdown()
{
  std::cout << "[head_motor_driver] Shutting down, holding last position" << std::endl;
  running_ = false;
  if (listen_fd_ >= 0) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
  }
}

namespace {
HeadMotorDriver* g_driver = nullptr;

void handleSignal(int)
{
  if (g_driver) g_driver->shutdown();
}
}  // namespace

int main()
{
  std::string stm32_host = "192.168.10.101";
  if (const char* env = std::getenv("HEAD_STM32_HOST")) stm32_host = env;

  uint16_t stm32_port = 9877;
  if (const char* env = std::getenv("HEAD_STM32_PORT")) stm32_port = static_cast<uint16_t>(std::stoi(env));

  std::string uds_path = "/tmp/openarm_head_motor.sock";
  if (const char* env = std::getenv("HEAD_UDS_PATH")) uds_path = env;

  std::string calibration_path =
    ament_index_cpp::get_package_share_directory("communication_devices") +
    "/config/head_calibration.yaml";
  if (const char* env = std::getenv("HEAD_CALIBRATION_PATH")) calibration_path = env;

  HeadMotorDriver driver(calibration_path, stm32_host, stm32_port, uds_path);
  g_driver = &driver;
  std::signal(SIGINT, handleSignal);
  std::signal(SIGTERM, handleSignal);

  std::thread uds_thread(&HeadMotorDriver::udsServerLoop, &driver);
  driver.controlLoop();
  uds_thread.join();

  return 0;
}
