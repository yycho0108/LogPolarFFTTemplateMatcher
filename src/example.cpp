#include <iostream>
#include <opencv2/highgui.hpp>

#include "fftm.hpp"

using namespace std;
using namespace cv;

static constexpr const int kImageSize = 512;
static constexpr const int kNumPos = 128;

int dx_dsc = kNumPos / 2, dy_dsc = kNumPos / 2, dr_dsc = kNumPos / 2;
float dx, dy, dr;

bool mdown = false;
float mx, my;

cv::Mat source, target, match;
std::vector<std::vector<cv::Point>> noise;

cv::Mat GetAffineMatrix(
    const cv::MatSize& size,
    const float dx, const float dy, const float dr)
{
    cv::Mat A = cv::Mat::zeros(2, 3, CV_32FC1);

    const float c = std::cos(dr);
    const float s = std::sin(dr);
    A.at<float>(0, 0) = c;
    A.at<float>(0, 1) = -s;
    A.at<float>(1, 0) = s;
    A.at<float>(1, 1) = c;

    const float cx = size[1] / 2;
    const float cy = size[0] / 2;

    A.at<float>(0, 2) = dx + (-c * cx + s * cy) + cx;
    A.at<float>(1, 2) = dy + (-s * cx - c * cy) + cy;

    return A;
}

static void UpdateTarget()
{
    // Create affine transform matrix.
    cv::Mat A = GetAffineMatrix(source.size, dx, dy, dr);

    //A = cv::getRotationMatrix2D(
    //    cv::Point2f(source.cols / 2, source.rows / 2),
    //    dr * (180 / M_PI),
    //    1.0f);

    source.copyTo(target);

    // Add noise.
    for (const auto& l : noise) {
        for (int i = 0; i < static_cast<int>(l.size()) - 1; ++i) {
            // std::cout << i << i + 1 << noise.size() << std::endl;
            const cv::Point& pa = l[i];
            const cv::Point& pb = l[i + 1];
            cv::line(target, pa, pb, cv::Scalar(255), 3);
        }
    }

    // Apply warp.
    cv::warpAffine(target, target, A, target.size());

    // Compute match.
    RotatedRect rr = LogPolarFFTTemplateMatch(source, target, 200, 100);
    cv::Mat A2 = GetAffineMatrix(source.size,
        rr.center.x - source.cols / 2,
        rr.center.y - source.rows / 2, (M_PI / 180) * rr.angle);
    cv::warpAffine(target, match, A2, match.size());

    // Plot rotated rectangle, to check result correctness.
    //source.copyTo(match);
    //Point2f rect_points[4];
    //rr.points(rect_points);
    //for (int j = 0; j < 4; j++) {
    //    line(match, rect_points[j], rect_points[(j + 1) % 4], Scalar(255, 0, 0), 2, cv::LINE_AA);
    //}

    cv::imshow("source", source);
    cv::imshow("target", target);
    cv::imshow("match", match);
}

static void OnDx(int, void*)
{
    dx = dx_dsc - 64;
    UpdateTarget();
}

static void OnDy(int, void*)
{
    dy = dy_dsc - 64;
    UpdateTarget();
}

static void OnDr(int, void*)
{
    dr = (dr_dsc - 64) * (M_PI / 64);
    std::cout << (dr * 180.0/ M_PI) << std::endl;
    UpdateTarget();
}

static void OnMouse(int event, int x, int y, int flags, void* param, cv::Mat* mat, bool draw)
{
    switch (event) {
    case cv::EVENT_LBUTTONDOWN: {
        mdown = true;
        if (draw) {
            mx = x;
            my = y;
        } else {
            cv::Mat A = GetAffineMatrix(target.size, dx, dy, dr);
            cv::Mat A2;
            cv::invertAffineTransform(A, A2);
            cv::Mat tmp = cv::Mat::zeros(2, 1, CV_32FC1);
            tmp.at<float>(0) = x;
            tmp.at<float>(1) = y;
            cv::Mat tmp2 = A2(cv::Rect(0, 0, 2, 2)) * tmp + A2(cv::Rect(2, 0, 1, 2));
            noise.emplace_back();
            noise.back().emplace_back(tmp2.at<float>(0),
                tmp2.at<float>(1));
        }
        break;
    }
    case cv::EVENT_MOUSEMOVE: {
        if (!mdown) {
            return;
        }
        if (draw) {
            cv::line(*mat, cv::Point(mx, my), cv::Point(x, y), cv::Scalar(255), 3);
        } else {
            cv::Mat A = GetAffineMatrix(target.size, dx, dy, dr);
            cv::Mat A2;
            cv::invertAffineTransform(A, A2);
            cv::Mat tmp = cv::Mat::zeros(2, 1, CV_32FC1);
            tmp.at<float>(0) = x;
            tmp.at<float>(1) = y;
            cv::Mat tmp2 = A2(cv::Rect(0, 0, 2, 2)) * tmp + A2(cv::Rect(2, 0, 1, 2));
            noise.back().emplace_back(tmp2.at<float>(0),
                tmp2.at<float>(1));
        }
        mx = x;
        my = y;
        cv::imshow((mat == &source) ? "source" : "target", *mat);
        break;
    }
    case cv::EVENT_LBUTTONUP: {
        mdown = false;
        break;
    }
    default:
        break;
    }
}

static void OnSourceMouse(int event, int x, int y, int flags, void* param)
{
    OnMouse(event, x, y, flags, param, &source, true);
}

static void OnTargetMouse(int event, int x, int y, int flags, void* param)
{
    OnMouse(event, x, y, flags, param, &target, false);
    UpdateTarget();
}

void SetupGui()
{
    // Source Image
    cv::namedWindow("source", cv::WINDOW_NORMAL);
    cv::imshow("source", source);

    // Target Image
    cv::namedWindow("target", cv::WINDOW_NORMAL);
    cv::imshow("target", target);

    // Match Image
    cv::namedWindow("match", cv::WINDOW_NORMAL);
    cv::imshow("match", match);

    // Control panel
    cv::namedWindow("control", cv::WINDOW_NORMAL);
    cv::createTrackbar("dx", "control", &dx_dsc, kNumPos, OnDx);
    cv::createTrackbar("dy", "control", &dy_dsc, kNumPos, OnDy);
    cv::createTrackbar("dr", "control", &dr_dsc, kNumPos, OnDr);

    cv::setMouseCallback("source", OnSourceMouse);
    cv::setMouseCallback("target", OnTargetMouse);

    cv::resizeWindow("source", 256, 256);
    cv::resizeWindow("target", 256, 256);
    cv::resizeWindow("match", 256, 256);
    cv::resizeWindow("control", 256, 256);

    cv::moveWindow("source", 0, 0);
    cv::moveWindow("target", 384, 0);
    cv::moveWindow("match", 0, 384);
    cv::moveWindow("control", 384, 384);
}

//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
int main(const int /*argc*/, const char** /*argv*/)
{
    // source = imread("cat.png", 1);
    source = cv::Mat::zeros(kImageSize, kImageSize, CV_8UC1);
    source.copyTo(target);
    source.copyTo(match);
    SetupGui();

    while (true) {
        const int k = cv::waitKey(1);
        switch (k) {
        case 27: {
            return 0;
        }
        default: {
            break;
        }
        }
    }
}
