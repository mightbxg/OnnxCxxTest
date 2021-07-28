#include "cvdnn_test.h"

#include <TestFuncs/TicToc.hpp>

using namespace std;

namespace cv {

void modelTest(const std::string& fn_model, const Mat& input, Mat& output,
    dnn::Backend backend, dnn::Target target)
{
    using namespace dnn;
    CV_Assert(input.type() == CV_8UC1);
    Net net = readNetFromONNX(fn_model);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    Mat blob;
    blobFromImage(input, blob, 1.0 / 255.0);

    net.setInput(blob);
    for (int i = 0; i < 10; ++i) {
        dbg::TicToc::ScopedTimer st("cvdnn", true);
        output = net.forward();
    }
    int new_size[] = { output.size[2], output.size[3] };
    output = output.reshape(1, 2, new_size);
    convertScaleAbs(output, output, 255.0);
}

} //namespace cv
