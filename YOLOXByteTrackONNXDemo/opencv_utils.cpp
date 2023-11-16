#include "opencv_utils.h"

/**
 * Implementation of draw_bboxes_opencv function.
 */
cv::Mat draw_bboxes_opencv(const cv::Mat& image, const std::vector<Eigen::VectorXf>& boxes, const std::vector<std::string>& labels,
    const std::vector<cv::Scalar>& colors, int width, int font_size, const std::vector<float>& probs) {

    // Clone the input image to avoid modifying the original image.
    cv::Mat annotated_image = image.clone();

    // Reference diagonal used for scaling font size.
    double REFERENCE_DIAGONAL = 1000;
    double diag = std::hypot(image.cols, image.rows);
    font_size = static_cast<int>(font_size * (diag / REFERENCE_DIAGONAL));

    // Font selection for the text.
    int cv_font = cv::FONT_HERSHEY_SIMPLEX;

    for (size_t i = 0; i < boxes.size(); ++i) {
        // Extracting bounding box coordinates.
        int x = static_cast<int>(boxes[i](0));
        int y = static_cast<int>(boxes[i](1));
        int w = static_cast<int>(boxes[i](2));
        int h = static_cast<int>(boxes[i](3));

        // Construct label string with probability if available.
        std::string label = labels[i];
        if (!probs.empty() && i < probs.size()) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << probs[i] * 100.0;
            label += ": " + oss.str() + "%";
        }

        // Drawing the bounding box.
        cv::rectangle(annotated_image, cv::Rect(x, y, w, h), colors[i], width);

        // Drawing the label background.
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv_font, font_size / 30.0, 1, &baseline);
        cv::rectangle(annotated_image, cv::Point(x, y - label_size.height - baseline),
            cv::Point(x + label_size.width, y), colors[i], cv::FILLED);

        // Calculating font color based on the mean color for visibility.
        double luminance = colors[i][2] * 0.299 + colors[i][1] * 0.587 + colors[i][0] * 0.114; // BGR order for OpenCV
        cv::Scalar font_color = luminance > 127.5 ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        // Drawing the label text.
        int font_thickness = 2;  // Adjust this value as per your requirement
        cv::putText(annotated_image, label, cv::Point(x, y), cv_font, font_size / 30.0, font_color, font_thickness);
    }

    return annotated_image;
}
