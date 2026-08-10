#pragma once
// -----------------------------------------------------------------------------
// hand_api_client.hpp
//
// Drives the 4-finger hand board through its standalone REST API (port 5051),
// not through ros2_control.
//
// That hand is not a ros2_control joint: it is a CH32V307 board on its own
// Ethernet subnet, and the only way to move it is to POST an angle per finger
// motor. The `hand_pose` step type talks to amazing_hand's
// FollowJointTrajectory controllers and does nothing on this hardware; the
// `hand_fingers` step type talks to this.
//
// ── Threading ────────────────────────────────────────────────────────────────
//
// The FSM runs on a single-threaded executor, so a blocking HTTP call here
// would freeze the whole state machine - including the cancel it would need to
// get out. So requests run on a worker thread and results are handed back
// through a queue that a timer drains on the executor thread. Callbacks
// therefore fire where every other FSM callback fires, and the FSM needs no
// locking of its own.
//
// Raw sockets rather than libcurl: this speaks fixed-shape JSON to one service
// on loopback. A hand-written HTTP/1.1 POST is about sixty lines and keeps a
// C++ dependency out of a workspace that otherwise has none.
// -----------------------------------------------------------------------------
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace sequence_executor {

// One motor of one finger. `finger` is 1-4 (F1..F4), `angle` is degrees.
// The board clamps each motor to its own safe range, so the applied value can
// differ from the requested one - that is the board's business, not an error.
struct FingerTarget
{
  int finger = 1;
  int motor = 1;      // 1 or 2
  double angle = 0.0;
};

class HandApiClient
{
public:
  using ResultCallback = std::function<void(bool success, const std::string& message)>;

  // `base_url` is host:port only, e.g. "127.0.0.1:5051".
  explicit HandApiClient(const rclcpp::Node::SharedPtr& node,
                         const std::string& host = "127.0.0.1", int port = 5051);
  ~HandApiClient();

  // Moves every target. The API takes one finger per call, so this issues them
  // in order on the worker thread and reports the first failure. `callback`
  // runs on the executor thread.
  void moveFingers(const std::vector<FingerTarget>& targets, ResultCallback callback);

  // Home all fingers, or one when `finger` is 1-4.
  void home(int finger, ResultCallback callback);

  // Cached answer to "is the board plugged in". Refreshed by probe(); returns
  // false until the first probe completes.
  bool boardReachable() const { return board_reachable_; }

  // Asks the API once, synchronously. Called during sequence validation, where
  // blocking briefly is fine because nothing is moving yet.
  bool probe(std::string& detail);

private:
  // Sends one request, returns false and fills `error` on any failure.
  bool request(const std::string& method, const std::string& path, const std::string& body,
               std::string& response, std::string& error);

  void enqueue(std::function<void()> work, ResultCallback callback);
  void drainCompletions();

  rclcpp::Node::SharedPtr node_;
  std::string host_;
  int port_;
  rclcpp::Logger logger_;

  bool board_reachable_ = false;

  // Worker side.
  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> pending_;
  std::deque<std::pair<ResultCallback, std::pair<bool, std::string>>> completed_;
  bool stop_ = false;

  // Executor side: drains `completed_` so callbacks land on the FSM's thread.
  rclcpp::TimerBase::SharedPtr completion_timer_;
};

}  // namespace sequence_executor
