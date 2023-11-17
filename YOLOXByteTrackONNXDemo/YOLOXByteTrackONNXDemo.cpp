#define NOMINMAX

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "OnnxRuntimeHelper.h"
#include "opencv_utils.h"
#include "yolox_utils.h"
#include "colormap_parser.h"
#include "BYTETracker.h"

/*
* Example usage: YOLOXByteTrackONNXDemo.exe hagrid-sample-30k-384p-yolox_tiny.onnx pexels-rodnae-productions-10373924.mp4 hagrid-sample-30k-384p-colormap.json
* Example usage: YOLOXByteTrackONNXDemo.exe hagrid-sample-30k-384p-yolox_tiny.onnx pexels-rodnae-productions-10373924.mp4 hagrid-sample-30k-384p-colormap.json Dml
*/

const int TARGET_SIZE = 288;
const int MAX_STRIDE = 32;
const float BBOX_CONF_THRESHOLD = 0.1f;
const float IOU_THRESH = 0.45f;
const int NUM_BBOX_Fields = 5;

int main(int argc, char** argv) {
    
    if (argc < 4 || argc > 5) {
        std::cout << "Usage: " << argv[0] << " [path_to_onnx_model] [path_to_input_video] [path_to_json_colormap] [execution_provider]" << std::endl;
        std::cout << "execution_provider is optional, defaults to 'CPU'. Options: 'CPU', 'Dml'" << std::endl;
        return -1;
    }

    // Set execution_provider based on the command line argument, default to "CPU"
    std::string execution_provider = (argc == 5) ? argv[4] : "CPU";

    // Ensure that execution_provider is either "CPU" or "Dml"
    if (execution_provider != "CPU" && execution_provider != "Dml") {
        std::cout << "Invalid execution provider. Must be 'CPU' or 'Dml'. Using default 'CPU'." << std::endl;
        execution_provider = "CPU";
    }


    // Intialize ONNX Runtime
    OnnxRuntimeHelper ort_helper;
    ort_helper.init_ort_api();
    std::cout << std::endl;
    std::cout << "Available Execution Providers: ";
    for (const auto& str : ort_helper.get_provider_names()) {
        std::cout << str << " ";
    }
    std::cout << std::endl;

    // Load the JSON colormap
    std::string colormap_filepath = argv[3];
    std::map<std::string, std::vector<int>> colormap_dict;
    std::vector<std::string> class_names;
    std::vector<std::vector<int>> int_colors;
    parse_colormap(colormap_filepath, colormap_dict, class_names, int_colors);
    
    // Initialize VideoCapture
    cv::VideoCapture cap(argv[2]);
    if (!cap.isOpened()) {
        std::cout << "Error: Could not open the video file." << std::endl;
        return -1;
    }   
    double frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);   // Get the width of the video
    double frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT); // Get the height of the video
    double fps = cap.get(cv::CAP_PROP_FPS);  // Get the FPS of the input video
    double total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << std::endl;
    std::cout << "Video Info:" << std::endl;
    std::cout << "Frame Width: " << frame_width << std::endl;
    std::cout << "Frame Height: " << frame_height << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "Frames: " << total_frames << std::endl;

    // Find the minimum and maximum dimension of the image
    int min_dim = std::min(frame_width, frame_height);
    int max_dim = std::max(frame_width, frame_height);
    double ratio = static_cast<double>(min_dim) / TARGET_SIZE;

    // Calculate new sizes based on ratio
    cv::Size resized_dims = (frame_width < frame_height)
        ? cv::Size(TARGET_SIZE, static_cast<int>(max_dim / ratio))
        : cv::Size(static_cast<int>(max_dim / ratio), TARGET_SIZE);
    float min_img_scale = (float)min_dim / std::min(resized_dims.width, resized_dims.height);
    
    auto align_to_32 = [](int dim) { return dim - dim % MAX_STRIDE; };
    cv::Size input_dims(align_to_32(resized_dims.width), align_to_32(resized_dims.height));
    
    // Calculate the offsets from the resized image dimensions to the input dimensions
    cv::Size offsets = (resized_dims - input_dims) / 2;
    Eigen::RowVector4f offsets_vector(offsets.width, offsets.height, 0, 0);
    cv::Rect crop_area(offsets.width, offsets.height, input_dims.width, input_dims.height);

    // Generate a matrix containing grid coordinates and strides for a given height and width.
    Eigen::MatrixXi output_grids = generate_output_grids(input_dims.height, input_dims.width);
    
    // # Load the model and create an InferenceSession
    ort_helper.load_model(argv[1], execution_provider.c_str(), input_dims);
    int proposal_length = class_names.size() + NUM_BBOX_Fields;
    const int output_length = output_grids.rows() * proposal_length;
    ort_helper.resize_output_data(output_length);
    
    // Initialize VideoWriter
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // Use MJPG codec.
    cv::Size frame_size(frame_width, frame_height);
    std::filesystem::path video_path(argv[2]);
    std::string video_out_path = video_path.stem().string() + "-out.mp4";
    cv::VideoWriter out(video_out_path, fourcc, fps, frame_size, true);

    // Initialize a ByteTracker object
    BYTETracker tracker(0.23, 30, 0.8, fps);    

    std::cout << std::endl;
    std::cout << "Processing Video Frames..." << std::endl;
    int frame_number = 0;
    cv::Mat frame;
    cv::Mat resized_img;
    cv::Mat input_img;

    // Initialize performance timers
    auto start = std::chrono::high_resolution_clock::now();
    auto start_inference = std::chrono::high_resolution_clock::now();
    auto end_inference = std::chrono::high_resolution_clock::now();
    double total_inference_duration = 0.0;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Prepare an input image for inference
        cv::resize(frame, resized_img, resized_dims);
        input_img = resized_img(crop_area);
        cv::cvtColor(input_img, input_img, cv::COLOR_BGR2RGB);

        // Start the inference timer
        start_inference = std::chrono::high_resolution_clock::now();

        // Perform inference
        ort_helper.perform_inference(input_img, output_length);

        // Process the model output
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mapped_output_float(ort_helper.get_output_data(), output_grids.rows(), proposal_length);
        Eigen::MatrixXd model_output = mapped_output_float.cast<double>();
        Eigen::MatrixXf proposals = process_output(mapped_output_float, output_grids, BBOX_CONF_THRESHOLD);
        proposals = filter_and_sort_proposals(proposals);

        // Apply non-max suppression to the proposals with the specified threshold
        Eigen::MatrixXf iou_matrix = calc_iou(proposals.leftCols(4));
        std::vector<int> indices_to_keep = nms_sorted_boxes(iou_matrix, IOU_THRESH);
        proposals = filter_proposals(proposals, indices_to_keep);

        // Compute the result using vectorized operations
        Eigen::MatrixXf bbox_list = (proposals.leftCols(4).rowwise() + offsets_vector) * min_img_scale;
        Eigen::MatrixXf label_list = proposals.col(4);
        Eigen::MatrixXf probs_list = proposals.col(5);

        // Update tracker with detections
        Eigen::MatrixXf boxes_probs(bbox_list.rows(), 5);
        boxes_probs << bbox_list, probs_list;
        std::vector<KalmanBBoxTrack>tracks = tracker.process_frame_detections(boxes_probs);
        Eigen::MatrixXf tlbr_boxes(bbox_list.rows(), 4);
        tlbr_boxes << bbox_list.col(0),          // top-left X
            bbox_list.col(1),                    // top-left Y
            bbox_list.col(0) + bbox_list.col(2), // bottom-right X
            bbox_list.col(1) + bbox_list.col(3); // bottom-right Y
        std::vector<int> track_ids_input(tlbr_boxes.rows(), -1);
        std::vector<int> track_ids = match_detections_with_tracks(tlbr_boxes.cast<double>(), track_ids_input, tracks);

        // Stop the inference timer after 'match_detections_with_tracks' call
        end_inference = std::chrono::high_resolution_clock::now();
        total_inference_duration += std::chrono::duration<double>(end_inference - start_inference).count();

        // Annotate the current frame with bounding boxes and tracking IDs
        std::vector<Eigen::VectorXf> filtered_bbox_list;
        std::vector<int> filtered_label_list;
        std::vector<float> filtered_probs_list;
        std::vector<int> filtered_track_ids;
        std::vector<std::string> labels;
        std::vector<cv::Scalar> colors;
        for (int i = 0; i < track_ids.size(); ++i) {
            if (track_ids[i] != -1) {
                filtered_bbox_list.push_back(bbox_list.row(i));
                filtered_label_list.push_back(label_list(i, 0));
                filtered_probs_list.push_back(probs_list(i, 0));
                filtered_track_ids.push_back(track_ids[i]);

                std::string label = class_names[(int)label_list(i, 0)];
                auto it = std::find(class_names.begin(), class_names.end(), label);
                int index = std::distance(class_names.begin(), it);
                cv::Scalar color(int_colors[index][2], int_colors[index][1], int_colors[index][0]);
                colors.push_back(color);
                
                std::ostringstream oss;
                oss << track_ids[i] << "-" << label;
                label = oss.str();
                labels.push_back(label);
            }
        }
        frame = draw_bboxes_opencv(frame, filtered_bbox_list, labels, colors, 2, 18, filtered_probs_list);

        // Write the frame to the output video
        out.write(frame);

        frame_number++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    // Calculating the difference in the times (in seconds)
    double duration_seconds = std::chrono::duration<double>(end - start).count();
    
    // Calculate the average iterations per second
    std::cout << "Average iterations per second: " << frame_number / duration_seconds << " it/s" << std::endl;

    // Calculate the average time spent in inference per frame
    double average_inference_time_per_frame = total_inference_duration / frame_number;

    // Output the average inference time per frame
    std::cout << "Average speed for inference and tracking per frame: " << 1/average_inference_time_per_frame << " FPS" << std::endl;

    cap.release();  // Release the VideoCapture
    out.release();  // Release the VideoWriter
    return 0;
}
