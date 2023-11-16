#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>

/**
 * @brief Draws bounding boxes on an image using OpenCV.
 *
 * @param image The source image on which to draw the bounding boxes.
 * @param boxes A vector of Eigen::VectorXf, each representing a bounding box as [x, y, width, height].
 * @param labels A vector of strings, each corresponding to the label for a bounding box.
 * @param colors A vector of cv::Scalar, each representing the color for a bounding box.
 * @param width The thickness of the bounding box lines.
 * @param font_size The base font size for text labels.
 * @param probs (Optional) A vector of probabilities corresponding to each bounding box.
 * @return cv::Mat The image with drawn bounding boxes.
 */
cv::Mat draw_bboxes_opencv(const cv::Mat& image, const std::vector<Eigen::VectorXf>& boxes,
    const std::vector<std::string>& labels, const std::vector<cv::Scalar>& colors,
    int width, int font_size, const std::vector<float>& probs = std::vector<float>());