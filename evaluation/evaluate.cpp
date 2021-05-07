#include <iostream>
#include <pybind11/embed.h>

namespace py = pybind11;

int main(int argc, char** argv) {

  py::scoped_interpreter guard{};
  py::object eval = py::module::import("evaluate_ate_scale");
  py::object evaluation = eval.attr("main");
  py::tuple results = evaluation("/home/pkok/thesis/ORB_SLAM3/evaluation/Ground_truth/EuRoC_left_cam/MH01_GT.txt", "/home/pkok/thesis/ORB_SLAM3/data/f_dataset-MH01_mono.txt");
  py::float_ ate_rmse = results[1];

  std::cout << (float) ate_rmse << std::endl;

  return EXIT_SUCCESS;
}
