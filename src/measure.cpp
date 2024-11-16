#include "measure.hpp"

#include <cassert>
#include <chrono>
#include <fstream>
#include <map>
#include <string>

namespace measure {
std::map<MeasurementEvent, std::string> __mes = {
    {MeasurementEvent::START, "START"}, {MeasurementEvent::END, "END"}};

std::string measure_event_name(MeasurementEvent me) { return __mes[me]; }

MeasurementPoint::MeasurementPoint(std::string func, MeasurementEvent event,
                                   time_point time)
    : func(func), event(event), time(time) {}

Interval::Interval(std::string func, double duration)
    : func(func), duration(duration) {}

Measure *Measure::instance = nullptr;

Measure *Measure::get_instance() {
  if (instance == nullptr) {
    instance = new Measure();
  }
  return instance;
}

Measure::Measure() : measurements({}) {
  measurements.reserve(25 * measurement_event_types.size());
}

void Measure::track_bytes(size_t bytes) { _bytes += bytes; }

void Measure::flush_bytes() {
  _bytes_measurements.push_back(_bytes);
  reset_bytes();
}

void Measure::reset_bytes() { _bytes = 0; }

size_t Measure::bytes() { return _bytes; }

void Measure::track(const std::string func, const MeasurementEvent event) {
  auto time = std::chrono::high_resolution_clock::now();
  auto point = MeasurementPoint(func, event, time);
  // This should never result in an extra allocation
  measurements.push_back(point);
}

std::vector<size_t> Measure::bytes_measurements() {
  return _bytes_measurements;
}

std::vector<Interval> Measure::intervals() {
  std::vector<Interval> intervals({});
  // This should be enough so that intervals is never re-allocated
  intervals.reserve(measurements.size() / 2);

  std::vector<MeasurementPoint> ms(measurements);
  while (ms.size() > 0) {
    // find the first starting point
    auto it = ms.begin();
    while (it != ms.end()) {
      if (it->event == MeasurementEvent::START)
        break;
      it = next(it);
    }
    // this means we didn't find a start event but we still have some
    // measurements. Ultimately, this points out that some measurement calls
    // are missing
    assert(it != ms.end());
    auto start_point = it;

    // find a matching end point
    it = next(it);
    while (it != ms.end()) {
      if (it->func == start_point->func && it->event == MeasurementEvent::END)
        break;
      it = next(it);
    }
    // this means we didn't find a matching end measurement. Ultimately, this
    // points out that some measurement calls are missing
    auto end_point = it;

    assert(end_point != ms.end());
    assert(start_point->func == end_point->func);

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end_point->time - start_point->time)
                        .count();
    auto interval = Interval(start_point->func, duration);
    intervals.push_back(interval);
    ms.erase(end_point);
    ms.erase(start_point);
  }

  return intervals;
}

void Measure::save(std::string path) {
  auto csv = std::ofstream(path);
  assert(!csv.fail());

  csv << "func,duration,bytes" << std::endl;
  auto bytes_data = Measure::get_instance()->bytes_measurements();
  for (auto i : bytes_data)
    csv << "bytes,0," << i << std::endl;

  auto data = Measure::get_instance()->intervals();
  for (auto i : data)
    csv << i.func << "," << i.duration << ",0" << std::endl;
}

void Measure::reset() {
  measurements.clear();
  _bytes = 0;
}
} // namespace measure