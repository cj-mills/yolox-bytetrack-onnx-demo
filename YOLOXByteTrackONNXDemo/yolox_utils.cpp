#include "yolox_utils.h"

/**
 * @brief Implement the generation of grid coordinates and strides for an image.
 *
 * The function calculates the grid coordinates (X, Y) and the corresponding stride
 * for each grid cell in the image. This information is crucial for translating
 * the model output into actual spatial locations in the image.
 *
 * @param height Height of the image.
 * @param width Width of the image.
 * @param strides Vector of strides.
 * @return Eigen::MatrixXi Matrix with each row containing (X, Y, stride).
 */
Eigen::MatrixXi generate_output_grids(int height, int width, const std::vector<int>& strides) {
    // Calculate total number of rows needed for the output matrix
    int total_rows = 0;
    for (int stride : strides) {
        total_rows += (height / stride) * (width / stride);
    }

    Eigen::MatrixXi output_grids(total_rows, 3);
    int row_offset = 0;

    for (int stride : strides) {
        int grid_height = height / stride;
        int grid_width = width / stride;
        int current_grid_size = grid_height * grid_width;

        // Create a matrix with 2 columns for X and Y coordinates using Eigen's reshaping
        Eigen::VectorXi vec_x = Eigen::VectorXi::LinSpaced(grid_width, 0, grid_width - 1);
        Eigen::VectorXi vec_y = Eigen::VectorXi::LinSpaced(grid_height, 0, grid_height - 1);

        Eigen::MatrixXi grid_x = vec_x.replicate(grid_height, 1);
        Eigen::MatrixXi grid_y = vec_y.replicate(1, grid_width).transpose();

        // Reshape and fill the output matrix
        output_grids.block(row_offset, 0, current_grid_size, 1) = Eigen::Map<Eigen::MatrixXi>(grid_x.data(), current_grid_size, 1);
        output_grids.block(row_offset, 1, current_grid_size, 1) = Eigen::Map<Eigen::MatrixXi>(grid_y.data(), current_grid_size, 1);
        output_grids.block(row_offset, 2, current_grid_size, 1).setConstant(stride);

        row_offset += current_grid_size;
    }

    return output_grids;
}


/**
 * @brief Process the raw output from a YOLOX model.
 *
 * Transforms the raw model output into interpretable object proposals, adjusting
 * bounding boxes and class probabilities. Applies a threshold to filter out
 * low-confidence predictions.
 *
 * @param model_output Raw model output matrix.
 * @param output_grids Grids and strides matrix.
 * @param bbox_conf_thresh Confidence threshold.
 * @return Eigen::MatrixXf Matrix of processed object proposals.
 */
Eigen::MatrixXf process_output(const Eigen::MatrixXf& model_output, const Eigen::MatrixXi& output_grids, float bbox_conf_thresh)
{
    int num_anchors = output_grids.rows();
    int proposal_length = model_output.cols();
    int num_classes = proposal_length - 5;

    Eigen::MatrixXf proposals(num_anchors, 7); // Each row: x0, y0, w, h, label, prob, valid_flag

    // Convert matrices to arrays for element-wise operations
    Eigen::ArrayXXf model_output_array = model_output.array();
    Eigen::ArrayXXi output_grids_array = output_grids.array();

    // Vectorized operations
    Eigen::ArrayXXf box_centroids = (model_output_array.block(0, 0, num_anchors, 2) + output_grids_array.block(0, 0, num_anchors, 2).cast<float>()) * output_grids_array.col(2).replicate(1, 2).cast<float>();

    Eigen::ArrayXf w = model_output_array.col(2).exp() * output_grids_array.col(2).cast<float>();
    Eigen::ArrayXf h = model_output_array.col(3).exp() * output_grids_array.col(2).cast<float>();

    Eigen::ArrayXf x0 = box_centroids.col(0) - w * 0.5f;
    Eigen::ArrayXf y0 = box_centroids.col(1) - h * 0.5f;

    Eigen::ArrayXXf class_probs = model_output_array.block(0, 5, num_anchors, num_classes);
    Eigen::ArrayXf box_objectness = model_output_array.col(4);
    Eigen::ArrayXXf final_probs = class_probs * box_objectness.replicate(1, num_classes);

    Eigen::VectorXf max_probs = final_probs.matrix().rowwise().maxCoeff();
    Eigen::VectorXi max_idxs = Eigen::VectorXi::Zero(num_anchors);
    for (int i = 0; i < num_anchors; ++i) {
        final_probs.row(i).matrix().maxCoeff(&max_idxs(i));
    }

    proposals << x0.matrix(), y0.matrix(), w.matrix(), h.matrix(), max_idxs.cast<float>(), max_probs, (max_probs.array() > bbox_conf_thresh).cast<float>().matrix();

    return proposals;
}

/**
 * @brief Filter and sort object proposals.
 *
 * Sorts the proposals based on confidence scores and filters out low-confidence ones.
 * This is used to prepare the proposals for further processing like NMS.
 *
 * @param proposals Proposals matrix.
 * @return Eigen::MatrixXf Sorted and filtered proposals matrix.
 */
Eigen::MatrixXf filter_and_sort_proposals(const Eigen::MatrixXf& proposals)
{
    int num_anchors = proposals.rows();

    // Sort the proposals based on the confidence score in descending order
    std::vector<int> sorted_indices(num_anchors);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
        [&proposals](int i1, int i2) {
            return proposals(i1, 5) > proposals(i2, 5);
        });

    Eigen::MatrixXf sorted_proposals(num_anchors, 7);
    for (int i = 0; i < num_anchors; ++i) {
        sorted_proposals.row(i) = proposals.row(sorted_indices[i]);
    }

    // Filter out rows with a validity flag of 0
    int valid_count = (sorted_proposals.col(6).array() == 1.0f).count();
    Eigen::MatrixXf valid_proposals = sorted_proposals.block(0, 0, valid_count, 7);
    valid_proposals.conservativeResize(valid_count, 6);

    return valid_proposals;
}

/**
 * @brief Calculate IoU for bounding box pairs.
 *
 * Computes the IoU for each pair of bounding boxes. This metric is vital for
 * determining the overlap between boxes and is used in NMS.
 *
 * @param proposals Bounding boxes matrix.
 * @return Eigen::MatrixXf IoU matrix.
 */
Eigen::MatrixXf calc_iou(const Eigen::MatrixXf& proposals) {
    int n = proposals.rows();
    Eigen::MatrixXf iou = Eigen::MatrixXf::Zero(n, n);

    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float x1 = std::max(proposals(i, 0), proposals(j, 0));
            float y1 = std::max(proposals(i, 1), proposals(j, 1));
            float x2 = std::min(proposals(i, 0) + proposals(i, 2), proposals(j, 0) + proposals(j, 2));
            float y2 = std::min(proposals(i, 1) + proposals(i, 3), proposals(j, 1) + proposals(j, 3));

            float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float union_ = proposals(i, 2) * proposals(i, 3) + proposals(j, 2) * proposals(j, 3) - intersection;

            if (union_ > 0) {
                iou(i, j) = intersection / union_;
                iou(j, i) = iou(i, j);
            }
        }
    }

    return iou;
}

/**
 * @brief Apply NMS to sorted bounding boxes.
 *
 * Filters out boxes that have high overlap with higher-scoring boxes.
 * This step is critical in removing redundant detections.
 *
 * @param iou IoU matrix.
 * @param iou_thresh IoU threshold for suppression.
 * @return std::vector<int> Indices of boxes to keep.
 */
std::vector<int> nms_sorted_boxes(const Eigen::MatrixXf& iou, float iou_thresh) {
    int n = iou.rows();
    std::vector<bool> mask(n, true);
    std::vector<int> keep;

    for (int i = 0; i < n; i++) {
        if (mask[i]) {
            keep.push_back(i);
            for (int j = i + 1; j < n; j++) {
                if (iou(i, j) > iou_thresh) {
                    mask[j] = false;
                }
            }
        }
    }

    return keep;
}

/**
 * @brief Filter proposals based on NMS results.
 *
 * Keeps only those proposals that are selected by NMS, effectively filtering
 * out overlapping and redundant detections.
 *
 * @param proposals Proposals matrix.
 * @param indices_to_keep Indices from NMS.
 * @return Eigen::MatrixXf Filtered proposals matrix.
 */
Eigen::MatrixXf filter_proposals(const Eigen::MatrixXf& proposals, const std::vector<int>& indices_to_keep) {
    Eigen::MatrixXf filtered_proposals(indices_to_keep.size(), proposals.cols());

    for (size_t i = 0; i < indices_to_keep.size(); i++) {
        filtered_proposals.row(i) = proposals.row(indices_to_keep[i]);
    }

    return filtered_proposals;
}