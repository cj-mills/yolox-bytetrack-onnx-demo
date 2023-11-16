#include "colormap_parser.h"

/**
 * @file colormap_parser.cpp
 *
 * @brief Implementation of the function to parse color map data from a JSON file.
 */

 /**
  * @brief Parses a colormap from a JSON file and stores the extracted information.
  *
  * @param filepath Path to the JSON file containing the colormap data.
  * @param colormap_dict Map associating class labels with RGB colors (as integer vectors).
  * @param class_names Vector to store the class labels found in the JSON file.
  * @param int_colors Vector of integer vectors to store the RGB values for each class.
  *
  * @details The function opens the specified JSON file and parses it to extract color
  * mapping information for each class label. The colors are stored as floating-point values
  * in the JSON and are converted to integers in the range [0, 255]. If the file can't be
  * opened or if the "items" key is not found in the JSON, it logs an error to `std::cerr`.
  * The function also handles JSON parsing exceptions.
  */
void parse_colormap(const std::string& filepath,
    std::map<std::string, std::vector<int>>& colormap_dict,
    std::vector<std::string>& class_names,
    std::vector<std::vector<int>>& int_colors) {

    // Read the JSON file
    std::ifstream colormap_file(filepath);
    if (!colormap_file.is_open()) {
        std::cerr << "Failed to open colormap file: " << filepath << std::endl;
        return;
    }

    nlohmann::json colormap_json;
    try {
        colormap_file >> colormap_json;
    }
    catch (const nlohmann::json::exception& e) {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;
        return;
    }

    // Check if "items" key exists
    if (colormap_json.find("items") == colormap_json.end()) {
        std::cerr << "\"items\" key not found in the JSON." << std::endl;
        return;
    }

    // Reserve space if possible to avoid reallocation
    const size_t num_items = colormap_json["items"].size();
    class_names.reserve(num_items);
    int_colors.reserve(num_items);

    for (const auto& item : colormap_json["items"]) {
        std::string label = item["label"];
        std::vector<int> color;
        color.reserve(item["color"].size());
        for (const auto& c : item["color"]) {
            color.emplace_back(static_cast<int>(c.get<double>() * 255));
        }

        colormap_dict[label] = color;
        class_names.emplace_back(label);
        int_colors.emplace_back(color);
    }
}
