#pragma once

#include <Eigen/Dense>
#include <vector>
#include <numeric>

/**
 * @brief Generate grid coordinates and strides for a given height and width.
 *
 * This function creates a matrix containing grid coordinates (X, Y) and corresponding
 * strides for an image of specific dimensions. This grid is essential for locating
 * objects in an image after processing through a neural network model.
 *
 * @param height Height of the image.
 * @param width Width of the image.
 * @param strides Vector of strides, with default values suitable for common YOLOX models.
 * @return Eigen::MatrixXi Matrix with each row containing (X, Y, stride).
 */
Eigen::MatrixXi generate_output_grids(int height, int width, const std::vector<int>& strides = { 8, 16, 32 });

/**
 * @brief Process the raw output from a YOLOX model.
 *
 * This function processes the raw output from a YOLOX model and transforms it into
 * a more interpretable format. It involves adjusting the bounding box predictions
 * with respect to grid coordinates and applying sigmoid to class probabilities.
 *
 * @param model_output Raw output from the YOLOX model.
 * @param output_grids Grid coordinates and strides generated for the image.
 * @param bbox_conf_thresh Threshold for filtering bounding box based on confidence scores.
 * @return Eigen::MatrixXf Matrix with each row representing a processed object proposal.
 */
Eigen::MatrixXf process_output(const Eigen::MatrixXf& model_output, const Eigen::MatrixXi& output_grids, float bbox_conf_thresh = 0.5);

/**
 * @brief Filter and sort the object proposals based on confidence scores.
 *
 * This function sorts the object proposals in descending order of their confidence scores
 * and then filters out proposals with low confidence scores.
 *
 * @param proposals Matrix of object proposals.
 * @return Eigen::MatrixXf Sorted and filtered matrix of object proposals.
 */
Eigen::MatrixXf filter_and_sort_proposals(const Eigen::MatrixXf& proposals);

/**
 * @brief Calculate the Intersection over Union (IoU) for all pairs of bounding boxes.
 *
 * IoU is a common metric used to measure the overlap between two bounding boxes.
 * This function calculates the IoU for each pair of bounding boxes in the proposals.
 *
 * @param proposals Matrix of bounding boxes (x, y, width, height).
 * @return Eigen::MatrixXf Matrix representing the IoU for each pair of bounding boxes.
 */
Eigen::MatrixXf calc_iou(const Eigen::MatrixXf& proposals);

/**
 * @brief Apply Non-Maximum Suppression (NMS) to the sorted bounding boxes.
 *
 * NMS is a technique to filter out overlapping bounding boxes, keeping only the ones
 * with the highest confidence scores. This function applies NMS based on the IoU threshold.
 *
 * @param iou IoU matrix for the bounding boxes.
 * @param iou_thresh Threshold for the IoU to apply NMS.
 * @return std::vector<int> Indices of the bounding boxes to keep.
 */
std::vector<int> nms_sorted_boxes(const Eigen::MatrixXf& iou, float iou_thresh);

/**
 * @brief Filter object proposals based on selected indices.
 *
 * After applying NMS, this function filters the original proposals, keeping only those
 * selected by NMS.
 *
 * @param proposals Matrix of object proposals.
 * @param indices_to_keep Indices of proposals to keep after NMS.
 * @return Eigen::MatrixXf Filtered matrix of object proposals.
 */
Eigen::MatrixXf filter_proposals(const Eigen::MatrixXf& proposals, const std::vector<int>& indices_to_keep);
