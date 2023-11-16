#include "OnnxRuntimeHelper.h"

// Constructor
OnnxRuntimeHelper::OnnxRuntimeHelper() {
    // Initialization code, if needed
    input_h = 0;
    input_w = 0;
    n_pixels = 0;
}

// Destructor
OnnxRuntimeHelper::~OnnxRuntimeHelper() {
    free_resources();
}

// Convert a standard string to a wide string, used for API calls requiring wide strings.
std::wstring OnnxRuntimeHelper::string_to_wstring(const std::string& str) {
    std::wstring wstr(str.begin(), str.end());
    return wstr;
}

// Initialize the ONNX Runtime API, retrieving the available providers.
void OnnxRuntimeHelper::init_ort_api() {
    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION); // Retrieve the ONNX Runtime API.

    char** raw_provider_names;
    int provider_count;

    ort->GetAvailableProviders(&raw_provider_names, &provider_count); // Fetch available providers.
    provider_names = std::vector<std::string>(raw_provider_names, raw_provider_names + provider_count); // Store them.
}

// Get the count of available providers.
int OnnxRuntimeHelper::get_provider_count() {
    return static_cast<int>(provider_names.size());
}

// Retrieve the name of an execution provider by its index.
const char* OnnxRuntimeHelper::get_provider_name(int index) {
    if (index >= 0 && index < provider_names.size()) {
        return provider_names[index].c_str();
    }
    return nullptr;
}

// Release all allocated resources, particularly the ONNX session and environment.
void OnnxRuntimeHelper::free_resources() {
    if (session) ort->ReleaseSession(session);
    if (env) ort->ReleaseEnv(env);
}

// Load an ONNX model for inference, specifying the model path, execution provider, and image dimensions.
const char* OnnxRuntimeHelper::load_model(const char* model_path, const char* execution_provider, cv::Size image_dims) {
    try {
        std::string instance_name = "inference-session";
        ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, instance_name.c_str(), &env);
        ort->DisableTelemetryEvents(env);
        ort->CreateSessionOptions(&session_options);

        std::string provider_name = execution_provider;
        std::unordered_map<std::string, std::function<void()>> execution_provider_actions = {
            {"CPU", []() {}},
            {"Dml", [&]() {   // Special settings for DirectML.
                ort->DisableMemPattern(session_options);
                ort->SetSessionExecutionMode(session_options, ExecutionMode::ORT_SEQUENTIAL);
                OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0);
            }}
        };

        bool action_taken = false;
        for (const auto& pair : execution_provider_actions) {
            if (provider_name.find(pair.first) != std::string::npos) {
                pair.second();
                action_taken = true;
                break;
            }
        }

        if (!action_taken) {
            return "Unknown execution provider specified.";
        }

        ort->CreateSession(env, string_to_wstring(model_path).c_str(), session_options, &session);
        ort->ReleaseSessionOptions(session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        char* temp_input_name;
        ort->SessionGetInputName(session, 0, allocator, &temp_input_name);
        input_name = temp_input_name;

        char* temp_output_name;
        ort->SessionGetOutputName(session, 0, allocator, &temp_output_name);
        output_name = temp_output_name;

        input_w = image_dims.width;
        input_h = image_dims.height;
        n_pixels = input_w * input_h;
        input_data.resize(n_pixels * n_channels);

        return "Model loaded successfully.";
    }
    catch (const std::exception& e) {
        return e.what();
    }
    catch (...) {
        return "An unknown error occurred while loading the model.";
    }
}

// Perform inference using the loaded ONNX model on the given image.
void OnnxRuntimeHelper::perform_inference(cv::Mat image, int length) {
    for (int p = 0; p < n_pixels; p++) {
        for (int ch = 0; ch < n_channels; ch++) {
            input_data[ch * n_pixels + p] = (image.data[p * n_channels + ch] / 255.0f);
        }
    }

    const char* input_names[] = { input_name.c_str() };
    const char* output_names[] = { output_name.c_str() };

    int64_t input_shape[] = { 1, 3, input_h, input_w };

    OrtMemoryInfo* memory_info;
    ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

    OrtValue* input_tensor = nullptr;
    ort->CreateTensorWithDataAsOrtValue(
        memory_info, input_data.data(), input_data.size() * sizeof(float),
        input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor
    );

    ort->ReleaseMemoryInfo(memory_info);

    OrtValue* output_tensor = nullptr;
    ort->Run(session, nullptr, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor);

    if (!output_tensor) {
        ort->ReleaseValue(input_tensor);
        return;
    }

    float* out_data;
    ort->GetTensorMutableData(output_tensor, (void**)&out_data);

    std::memcpy(output_data.data(), out_data, length * sizeof(float));

    ort->ReleaseValue(input_tensor);
    ort->ReleaseValue(output_tensor);
}

// Accessor methods for provider names and image dimensions.
const std::vector<std::string>& OnnxRuntimeHelper::get_provider_names() const {
    return provider_names;
}

int OnnxRuntimeHelper::get_input_height() const {
    return input_h;
}

int OnnxRuntimeHelper::get_input_width() const {
    return input_w;
}

int OnnxRuntimeHelper::get_pixel_count() const {
    return n_pixels;
}

// Resize the output data buffer to a specified length.
void OnnxRuntimeHelper::resize_output_data(int output_length) {
    output_data.resize(output_length);
}

// Accessor methods for output data buffer.
float* OnnxRuntimeHelper::get_output_data() {
    return output_data.data();
}

const float* OnnxRuntimeHelper::get_output_data() const {
    return output_data.data();
}
