# YOLOX ByteTrack-Eigen ONNX Demo
A Visual Studio project demonstrating how to perform object tracking across video frames with YOLOX, ONNX Runtime, and the [ByteTrack-Eigen](https://github.com/cj-mills/byte-track-eigen) library.



## Setup Steps:

1. Clone the repository.
2. Run the `download-dependencies.bat` setup file.
3. Ensure the Solution Configuration for the Visual Studio project is in `Release` mode.
4. Restore the NuGet packages.
5. Build the solution (`Ctrl+Shift+B`).
6. Place test files in the folder with the executable (sample files available below).



## Example Usage:

### Sample Files

* [pexels-rodnae-productions-10373924.mp4](https://huggingface.co/datasets/cj-mills/pexels-object-tracking-test-videos/resolve/main/pexels-rodnae-productions-10373924.mp4?download=true)
* [hagrid-sample-30k-384p-yolox_tiny.onnx](https://huggingface.co/cj-mills/yolox-hagrid-onnx/resolve/main/yolox_tiny/hagrid-sample-30k-384p-yolox_tiny.onnx?download=true)
* [hagrid-sample-30k-384p-colormap.json](https://huggingface.co/cj-mills/yolox-hagrid-onnx/resolve/main/hagrid-sample-30k-384p-colormap.json?download=true)
  * (To save the colormap file, right-click the link and opt for `Save Link As...`)

### CPU Inference

```bash
YOLOXByteTrackONNXDemo.exe hagrid-sample-30k-384p-yolox_tiny.onnx pexels-rodnae-productions-10373924.mp4 hagrid-sample-30k-384p-colormap.json
```

### DirectML Inference

```bash
YOLOXByteTrackONNXDemo.exe hagrid-sample-30k-384p-yolox_tiny.onnx pexels-rodnae-productions-10373924.mp4 hagrid-sample-30k-384p-colormap.json Dml
```

