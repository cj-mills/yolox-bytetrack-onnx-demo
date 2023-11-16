#pragma once

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

/**
 * @file colormap_parser.h
 *
 * @brief Defines the function to parse color map data from a JSON file.
 *
 * This file declares the function `parse_colormap` which reads a JSON file to extract
 * color mapping information. Each color is represented as an RGB value in a floating-point
 * format and is converted to an integer format.
 */

 /**
  * @brief Parses a colormap from a JSON file and stores the extracted information.
  *
  * @param filepath Path to the JSON file containing the colormap data.
  * @param colormap_dict Map associating class labels with RGB colors (as integer vectors).
  * @param class_names Vector to store the class labels found in the JSON file.
  * @param int_colors Vector of integer vectors to store the RGB values for each class.
  *
  * @details This function reads a JSON file specified by `filepath` and parses it to
  * extract color mapping information. The color values are expected to be in a floating-point
  * format and are converted to integers in the range [0, 255]. It updates `colormap_dict`,
  * `class_names`, and `int_colors` with the parsed data. The function handles file opening
  * errors and JSON parsing exceptions.
  */
void parse_colormap(const std::string& filepath,
    std::map<std::string, std::vector<int>>& colormap_dict,
    std::vector<std::string>& class_names,
    std::vector<std::vector<int>>& int_colors);
