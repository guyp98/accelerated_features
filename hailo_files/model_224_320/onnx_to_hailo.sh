
hailo parser onnx x_feature_13_without_pixel_unshuffle_normilize_softmax_slice_224_320_sim.onnx --start-node-names /block1/block1.0/layer/layer.0/Conv /keypoint_head/keypoint_head.0/layer/layer.0/Conv /skip1/skip1.0/AveragePool --end-node-names OUTPUT_1 OUTPUT_2 OUTPUT_3
hailo optimize x_feature_13_without_pixel_unshuffle_normilize_softmax_slice_224_320_sim.har --calib-set-path ../calibsets/combined_array_transposed_with_onnx_head_224_320.npy 
hailo compiler x_feature_13_without_pixel_unshuffle_normilize_softmax_slice_224_320_sim_optimized.har 

python3 split_onnx.py --onnx-path x_feature_13_without_pixel_unshuffle_normilize_softmax_slice_224_320_sim.onnx --onnx-output x_feature_13_without_pixel_unshuffle_normilize_softmax_slice_224_320_sim_only_head.onnx --subgraph-output /norm/InstanceNormalization_output_0  --subgraph-input INPUT_1