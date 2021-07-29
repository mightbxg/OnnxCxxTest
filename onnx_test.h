#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

namespace Ort {

void modelTest(const std::string& fn_model, const cv::Mat& input, cv::Mat& output, bool use_cuda);

} //namespace Ort
