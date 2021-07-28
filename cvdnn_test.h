#pragma once

#include <opencv2/opencv.hpp>

namespace cv {

void modelTest(const std::string& fn_model, const Mat& input, Mat& output,
    dnn::Backend backend, dnn::Target target);

} //namespace cv
