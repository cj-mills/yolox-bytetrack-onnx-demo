# YOLOX ByteTrack-Eigen ONNX Demo
A Visual Studio project demonstrating how to perform object tracking across video frames with YOLOX, ONNX Runtime, and the ByteTrack-Eigen library.



## Setup Steps:

1. Run the `download-dependencies.bat` setup file.
2. Ensure the Solution Configuration for the Visual Studio project is in `Release` mode.
3. Restore the NuGet packages.





## Example Usage:

### CPU Inference

```bash
YOLOXByteTrackONNXDemo.exe hagrid-sample-30k-384p-yolox_tiny.onnx pexels-rodnae-productions-10373924.mp4 hagrid-sample-30k-384p-colormap.json
```



### DirectML Inference

```bash
YOLOXByteTrackONNXDemo.exe hagrid-sample-30k-384p-yolox_tiny.onnx pexels-rodnae-productions-10373924.mp4 hagrid-sample-30k-384p-colormap.json Dml
```

