#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#define measure_point(measurement_func, measurement_type)                      \
  measure::Measure::get_instance()->track(measurement_func, measurement_type)

namespace measure {

typedef std::chrono::high_resolution_clock::time_point time_point;

static const std::string gemm = "gemm";
static const std::string global = "global";

enum class MeasurementEvent : uint8_t { START, END };
const std::array<MeasurementEvent, 2> measurement_event_types = {
    MeasurementEvent::START, MeasurementEvent::END};
std::string measurement_event_name(MeasurementEvent me);

class MeasurementPoint {
public:
  std::string func;
  MeasurementEvent event;
  time_point time;

  MeasurementPoint(std::string func, MeasurementEvent event, time_point time);
};

class Interval {
public:
  std::string func;
  double duration; // nanoseconds

  Interval(std::string func, double duration);
};

class Measure {
private:
  static Measure *instance;
  std::vector<MeasurementPoint> measurements;
  size_t _bytes;

public:
  Measure();
  static Measure *get_instance();

  void track(const std::string func, const MeasurementEvent event);
  void track_bytes(size_t bytes);
  std::vector<Interval> intervals();
  void save(const std::string path);
  size_t bytes();
  void reset();
};
} // namespace measure
