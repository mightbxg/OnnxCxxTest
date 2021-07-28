#pragma once

#include <opencv2/opencv.hpp>
#include <torch/script.h>

namespace torch {

Tensor toTensor(const cv::Mat& image);
cv::Mat toMat(const Tensor& tensor);

void torchTest(const string& fn_model, const Tensor& input, Tensor& output, bool use_cuda);

} //namespace torch
