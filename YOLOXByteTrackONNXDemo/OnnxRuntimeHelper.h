#pragma once

#include <onnxruntime_cxx_api.h>  // ONNX Runtime C++ API header for model inference.
#include "dml_provider_factory.h" // DirectML provider factory header for hardware acceleration.
#include <opencv2/opencv.hpp>     // OpenCV header for image processing.
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>

/**
 * @brief Helper class for interfacing with ONNX Runtime.
 *
 * This class provides a simplified and structured way to use the ONNX Runtime
 * for performing machine learning model inference, particularly focusing on
 * image data. It encapsulates initialization, model loading, and inference
 * execution, offering an easier interface for clients.
 */
class OnnxRuntimeHelper {
private:
    int input_w; // Width of the input image.
    int input_h; // Height of the input image.
    int n_pixels; // Total number of pixels in the input image.
    const int n_channels = 3; // Number of color channels in the input image (assumed to be 3 for RGB).
    const OrtApi* ort = nullptr; // Pointer to ONNX Runtime API.
    std::vector<std::string> provider_names; // List of available execution providers.
    OrtEnv* env = nullptr; // ONNX Runtime environment.
    OrtSessionOptions* session_options = nullptr; // Session options for the ONNX Runtime session.
    OrtSession* session = nullptr; // ONNX Runtime session for model execution.
    std::string input_name; // Name of the input node in the ONNX model.
    std::string output_name; // Name of the output node in the ONNX model.
    std::vector<float> input_data; // Buffer for input data to the model.
    std::vector<float> output_data; // Buffer for output data from the model.

    // Converts a standard string to a wide string.
    std::wstring string_to_wstring(const std::string& str);

public:
    // Constructor.
    OnnxRuntimeHelper();

    // Destructor.
    ~OnnxRuntimeHelper();

    // Initialize the ONNX Runtime API.
    void init_ort_api();

    // Get the number of available execution providers.
    int get_provider_count();

    // Get the name of an execution provider by index.
    const char* get_provider_name(int index);

    // Release all allocated resources.
    void free_resources();

    /**
     * Load an ONNX model and prepare it for inference.
     *
     * @param model_path Path to the ONNX model file.
     * @param execution_provider Name of the execution provider to use.
     * @param image_dims Dimensions of the input image.
     * @return A pointer to a constant character string detailing the status.
     */
    const char* load_model(const char* model_path, const char* execution_provider, cv::Size image_dims);

    /**
     * Perform inference using the loaded ONNX model.
     *
     * @param image Image to be processed.
     * @param length Length of the output data.
     */
    void perform_inference(cv::Mat image, int length);

    // Get the names of the available execution providers.
    const std::vector<std::string>& get_provider_names() const;

    // Get the height of the input image.
    int get_input_height() const;

    // Get the width of the input image.
    int get_input_width() const;

    // Get the total pixel count of the input image.
    int get_pixel_count() const;

    // Resize the output data buffer.
    void resize_output_data(int output_length);

    // Get a pointer to the output data buffer.
    float* get_output_data();

    // Get a constant pointer to the output data buffer.
    const float* get_output_data() const;
};
