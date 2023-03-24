crate::ix!();

#[test] fn bound_shape_inference_sparse_lengths_sum() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "SparseLengthsSum", "", {"Weights", "Data", "Lengths"}, {"Out"}, {}));
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "Weights",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {1000, 16}));
      BoundShapeSpec spec(20, 1000);
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "Weights",
          {TensorBoundShape_DimType_CONSTANT, TensorBoundShape_DimType_CONSTANT},
          {1000, 16});
      verifyShapeInfo(
          out_shape,
          "Data",
          {TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT},
          {spec.max_batch_size * spec.max_seq_size},
          TensorProto_DataType_INT64);
      verifyShapeInfo(
          out_shape,
          "Lengths",
          {TensorBoundShape_DimType_BATCH},
          {spec.max_batch_size},
          TensorProto_DataType_INT32);
      verifyShapeInfo(
          out_shape,
          "Out",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 16});
  */
}


#[test] fn bound_shape_inference_sparse_lengths_sum_sparse_lookup() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "SparseLengthsSumSparseLookup",
          "",
          {"Indices", "Lengths", "Remapping", "Weights"},
          {"IndicesOut", "LengthsOut", "WeightsOut"},
          {}));
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "Remapping",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT},
              {1000},
              TensorProto_DataType_INT32));
      BoundShapeSpec spec(20, 1000);
      shape_map.emplace(
          "Indices",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT},
              {spec.max_batch_size * spec.max_seq_size},
              TensorProto_DataType_INT32));
      shape_map.emplace(
          "Weights",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT},
              {spec.max_batch_size * spec.max_seq_size},
              TensorProto_DataType_FLOAT));
      shape_map.emplace(
          "Lengths",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH},
              {spec.max_batch_size},
              TensorProto_DataType_INT32));
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "WeightsOut",
          {TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT},
          {spec.max_batch_size * spec.max_seq_size},
          TensorProto_DataType_FLOAT);
      verifyShapeInfo(
          out_shape,
          "IndicesOut",
          {TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT},
          {spec.max_batch_size * spec.max_seq_size},
          TensorProto_DataType_INT32);
      verifyShapeInfo(
          out_shape,
          "LengthsOut",
          {TensorBoundShape_DimType_BATCH},
          {spec.max_batch_size},
          TensorProto_DataType_INT32);
  */
}


#[test] fn bound_shape_inference_sparse_lengths_sum_fused_8bit_rowwise() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "SparseLengthsSumFused8BitRowwise",
          "",
          {"Weights", "Data", "Lengths"},
          {"Out"},
          {}));
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "Weights",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {1000, 58},
              TensorProto_DataType_INT8));
      BoundShapeSpec spec(20, 1000);
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "Weights",
          {TensorBoundShape_DimType_CONSTANT, TensorBoundShape_DimType_CONSTANT},
          {1000, 58},
          TensorProto_DataType_INT8);
      verifyShapeInfo(
          out_shape,
          "Data",
          {TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT},
          {spec.max_batch_size * spec.max_seq_size},
          TensorProto_DataType_INT64);
      verifyShapeInfo(
          out_shape,
          "Lengths",
          {TensorBoundShape_DimType_BATCH},
          {spec.max_batch_size},
          TensorProto_DataType_INT32);
      verifyShapeInfo(
          out_shape,
          "Out",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 50});
  */
}


#[test] fn bound_shape_inference_sparse_lengths_sum_8bit_rowwise_sparse() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "SparseLengthsSum8BitRowwiseSparse",
          "",
          {"Weights", "Data", "Lengths", "Mapping"},
          {"Out"},
          {}));
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "Weights",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {1000, 58},
              TensorProto_DataType_INT8));
      shape_map.emplace(
          "Mapping",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT},
              {2000},
              TensorProto_DataType_INT32));
      BoundShapeSpec spec(20, 1000);
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "Weights",
          {TensorBoundShape_DimType_CONSTANT, TensorBoundShape_DimType_CONSTANT},
          {1000, 58},
          TensorProto_DataType_INT8);
      verifyShapeInfo(
          out_shape,
          "Mapping",
          {TensorBoundShape_DimType_CONSTANT},
          {2000},
          TensorProto_DataType_INT32);
      verifyShapeInfo(
          out_shape,
          "Data",
          {TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT},
          {spec.max_batch_size * spec.max_seq_size},
          TensorProto_DataType_INT64);
      verifyShapeInfo(
          out_shape,
          "Lengths",
          {TensorBoundShape_DimType_BATCH},
          {spec.max_batch_size},
          TensorProto_DataType_INT32);
      verifyShapeInfo(
          out_shape,
          "Out",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 50});
  */
}


#[test] fn bound_shape_inference_sparse_lengths_sum_fused_4bit_rowwise() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "SparseLengthsSumFused4BitRowwise",
          "",
          {"Weights", "Data", "Lengths"},
          {"Out"},
          {}));
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "Weights",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {1000, 54},
              TensorProto_DataType_INT8));
      BoundShapeSpec spec(20, 1000);
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "Weights",
          {TensorBoundShape_DimType_CONSTANT, TensorBoundShape_DimType_CONSTANT},
          {1000, 54},
          TensorProto_DataType_INT8);
      verifyShapeInfo(
          out_shape,
          "Data",
          {TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT},
          {spec.max_batch_size * spec.max_seq_size},
          TensorProto_DataType_INT64);
      verifyShapeInfo(
          out_shape,
          "Lengths",
          {TensorBoundShape_DimType_BATCH},
          {spec.max_batch_size},
          TensorProto_DataType_INT32);
      verifyShapeInfo(
          out_shape,
          "Out",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 100});
  */
}


#[test] fn bound_shape_inference_lengths_range_fill() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(
          CreateOperatorDef("LengthsRangeFill", "", {"X"}, {"Y"}, {}));
      net.add_op()->CopyFrom(CreateOperatorDef("Copy", "", {"Y"}, {"Z"}, {}));
      ShapeInfoMap shape_map;
      BoundShapeSpec spec(20, 1000);
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "X",
          {TensorBoundShape_DimType_BATCH},
          {spec.max_batch_size},
          TensorProto_DataType_INT32);
      verifyShapeInfo(
          out_shape,
          "Y",
          {TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT},
          {spec.max_batch_size * spec.max_seq_size},
          TensorProto_DataType_INT32);
      verifyShapeInfo(
          out_shape,
          "Z",
          {TensorBoundShape_DimType_BATCH_OF_FEATURE_MAX_DEFAULT},
          {spec.max_batch_size * spec.max_seq_size},
          TensorProto_DataType_INT32);
  */
}



#[test] fn bound_shape_inference_constant_fill() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(
          CreateOperatorDef("ConstantFill", "", {"X"}, {"Y"}, {}));
      ShapeInfoMap shape_map;
      BoundShapeSpec spec(20, 1000);
      BoundShapeInferencer eng(spec);
      shape_map.emplace(
          "X",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH,
               TensorBoundShape_DimType_CONSTANT},
              {20, 1024}));
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "Y",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {20, 1024},
          TensorProto_DataType_FLOAT);
  */
}


// https://github.com/pytorch/pytorch/issues/40861
#[cfg(not(target_os = "windows"))]
#[test] fn bound_shape_inference_reshape() {
    todo!();
    /*
      NetDef net;
      std::vector<int> new_shape{-1, 8};
      std::vector<int> new_shape2{2, 8};
      net.add_op()->CopyFrom(
          CreateOperatorDef("FC", "", {"X0", "W0", "B0"}, {"X1"}, {}));
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Reshape",
          "",
          {"X1"},
          {"Y1", "old_shape"},
          {MakeArgument<std::vector<int>>("shape", new_shape)}));

      // Cannot infer shape for this one because input/output shape doesn't match
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Reshape",
          "",
          {"X1"},
          {"Y2", "old_shape2"},
          {MakeArgument<std::vector<int>>("shape", new_shape2)}));
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "W0",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {16, 1024}));
      shape_map.emplace(
          "B0", makeTensorInfo({TensorBoundShape_DimType_CONSTANT}, {16}));
      BoundShapeSpec spec(20, 1000);
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "X0",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 1024});
      verifyShapeInfo(
          out_shape,
          "X1",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 16});
      verifyShapeInfo(
          out_shape,
          "Y1",
          {TensorBoundShape_DimType_BATCH,
           TensorBoundShape_DimType_CONSTANT}, // TODO
          {spec.max_batch_size * 16 / 8, 8});
      EXPECT_TRUE(out_shape.find("Y2") == out_shape.end());
  */
}


#[test] fn bound_shape_inference_concat_missing_input() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Concat",
          "",
          {"I0", "I1"},
          {"Cout", "split_info"},
          {MakeArgument<int>("axis", 1), MakeArgument<int>("add_axis", 1)}));
      BoundShapeSpec spec(20, 1000);
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "I0",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
              {spec.max_batch_size, 60}));
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "I0",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 60});
      verifyShapeInfo(
          out_shape,
          "Cout",
          {TensorBoundShape_DimType_BATCH,
           TensorBoundShape_DimType_CONSTANT,
           TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 2, 60});
  */
}


// See https://github.com/pytorch/pytorch/issues/35544
#[test] fn bound_shape_inference_int_8quantize_infer_input_backwards() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Int8Quantize",
          "",
          {"I0"},
          {"Cout", "split_info"},
          {MakeArgument<int>("Y_zero_point", 0),
           MakeArgument<float>("Y_scale", 0.05)}));
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Int8FC",
          "",
          {"Cout", "W0", "B0"},
          {"Y"},
          {MakeArgument<int>("Y_zero_point", 0),
           MakeArgument<float>("Y_scale", 0.05)}));
      BoundShapeSpec spec(20, 1000);
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "W0",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {16, 101},
              TensorProto_DataType_INT8,
              true));
      shape_map.emplace(
          "B0",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT},
              {16},
              TensorProto_DataType_INT32,
              true));
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "I0",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 101});
      verifyShapeInfo(
          out_shape,
          "Cout",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 101},
          TensorProto_DataType_UINT8,
          true);
      verifyShapeInfo(
          out_shape,
          "Y",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 16},
          TensorProto_DataType_UINT8,
          true);
  */
}


#[test] fn bound_shape_inference_concat_infer_input_backwards() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Concat",
          "",
          {"I0", "I1"},
          {"Cout", "split_info"},
          {MakeArgument<int>("axis", 1)}));
      net.add_op()->CopyFrom(
          CreateOperatorDef("FCTransposed", "", {"Cout", "W0", "B0"}, {"Y"}, {}));
      BoundShapeSpec spec(20, 1000);
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "I0",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
              {spec.max_batch_size, 60}));
      shape_map.emplace(
          "W0",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {101, 16}));
      shape_map.emplace(
          "B0", makeTensorInfo({TensorBoundShape_DimType_CONSTANT}, {16}));
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "I0",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 60});
      verifyShapeInfo(
          out_shape,
          "Cout",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 101});
      verifyShapeInfo(
          out_shape,
          "Y",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 16});
      verifyShapeInfo(
          out_shape,
          "I1",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 101 - 60});
  */
}


#[test] fn bound_shape_inference_elementwise_infer_input_backwards() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Mul", "", {"I0", "I1"}, {"Out"}, {MakeArgument<int>("broadcast", 1)}));
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Add",
          "",
          {"I00", "I11"},
          {"Outt"},
          {MakeArgument<int>("broadcast", 1)}));
      BoundShapeSpec spec(20, 1000);
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "Out",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
              {spec.max_batch_size, 60}));
      shape_map.emplace(
          "Outt",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
              {spec.max_batch_size, 50}));
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "I0",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 60});
      verifyShapeInfo(
          out_shape,
          "I1",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 60});
      verifyShapeInfo(
          out_shape,
          "I00",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 50});
      verifyShapeInfo(
          out_shape,
          "I11",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 50});
  */
}


#[test] fn bound_shape_inference_elementwise_op() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Mul", "", {"I0", "I1"}, {"Out"}, {MakeArgument<int>("broadcast", 1)}));
      net.add_op()->CopyFrom(CreateOperatorDef("Mul", "", {"I3", "I4"}, {"Out3"}));
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Add",
          "",
          {"I00", "I11"},
          {"Outt"},
          {MakeArgument<int>("broadcast", 1)}));
      BoundShapeSpec spec(20, 1000);
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "I0",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
              {spec.max_batch_size, 60}));
      shape_map.emplace(
          "I00",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
              {spec.max_batch_size, 50}));
      shape_map.emplace(
          "I3",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
              {spec.max_batch_size, 40}));
      shape_map.emplace(
          "I4",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
              {spec.max_batch_size, 40}));
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "I1",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 60});
      verifyShapeInfo(
          out_shape,
          "Out",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 60});
      verifyShapeInfo(
          out_shape,
          "I11",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 50});
      verifyShapeInfo(
          out_shape,
          "Outt",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 50});
      verifyShapeInfo(
          out_shape,
          "Out3",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 40});
  */
}


#[test] fn bound_shape_inference_bucketize() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Bucketize",
          "",
          {"In"},
          {"Out"},
          {MakeArgument<std::vector<float>>("boundaries", {1.0, 2.0})}));
      BoundShapeSpec spec(20, 1000);
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "In",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
              {spec.max_batch_size, 60}));
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "Out",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 60},
          TensorProto_DataType_INT32);
  */
}


#[test] fn bound_shape_inference_split() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Split", "", {"X"}, {"Y0", "Y1"}, {MakeArgument<int>("axis", 1)}));
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Split",
          "",
          {"X"},
          {"Y2", "Y3", "Y4"},
          {MakeArgument<int>("axis", 1),
           MakeArgument<std::vector<int>>("split", {4, 30, 14})}));
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Split",
          "",
          {"X1"},
          {"Y5", "Y6"},
          {MakeArgument<int>("axis", 1), MakeArgument<int>("add_axis", 1)}));
      BoundShapeSpec spec(20, 1000);
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "X",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
              {spec.max_batch_size, 48}));
      shape_map.emplace(
          "X1",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH,
               TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {spec.max_batch_size, 2, 48}));
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "X",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 48});
      verifyShapeInfo(
          out_shape,
          "X1",
          {TensorBoundShape_DimType_BATCH,
           TensorBoundShape_DimType_CONSTANT,
           TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 2, 48});
      verifyShapeInfo(
          out_shape,
          "Y0",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 48 / 2});
      verifyShapeInfo(
          out_shape,
          "Y1",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 48 / 2});
      verifyShapeInfo(
          out_shape,
          "Y2",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 4});
      verifyShapeInfo(
          out_shape,
          "Y3",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 30});
      verifyShapeInfo(
          out_shape,
          "Y4",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 14});
      verifyShapeInfo(
          out_shape,
          "Y5",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 48});
      verifyShapeInfo(
          out_shape,
          "Y6",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 48});
  */
}


// https://github.com/pytorch/pytorch/issues/41471
#[test] fn bound_shape_inference_fc() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(
          CreateOperatorDef("FC", "", {"X0", "W0", "B0"}, {"Out0"}, {}));
      net.add_op()->CopyFrom(
          CreateOperatorDef("FCTransposed", "", {"X1", "W1", "B1"}, {"Out1"}, {}));
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Int8FC", "", {"X2", "W2", "B2", "quant_param"}, {"Out2"}, {}));
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "W0",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {16, 1024}));
      shape_map.emplace(
          "B0", makeTensorInfo({TensorBoundShape_DimType_CONSTANT}, {16}));
      shape_map.emplace(
          "W1",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {16, 1024}));
      shape_map.emplace(
          "B1", makeTensorInfo({TensorBoundShape_DimType_CONSTANT}, {1024}));

      shape_map.emplace(
          "W2",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {16, 1024}));
      shape_map.emplace(
          "B2", makeTensorInfo({TensorBoundShape_DimType_CONSTANT}, {16}));
      shape_map.emplace(
          "quant_param", makeTensorInfo({TensorBoundShape_DimType_CONSTANT}, {1}));

      BoundShapeSpec spec(20, 1000);
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "X0",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 1024});
      verifyShapeInfo(
          out_shape,
          "Out0",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 16});
      verifyShapeInfo(
          out_shape,
          "X1",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 16});
      verifyShapeInfo(
          out_shape,
          "Out1",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 1024});
      verifyShapeInfo(
          out_shape,
          "X2",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 1024},
          TensorProto_DataType_UINT8,
          true);
      verifyShapeInfo(
          out_shape,
          "Out2",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 16},
          TensorProto_DataType_UINT8,
          true);
  */
}


#[test] fn bound_shape_inference_fc3d() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(
          CreateOperatorDef("FC", "", {"X0", "W0", "B0"}, {"Out0"}, {}));
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "W0",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {16, 1, 1024}));
      shape_map.emplace(
          "B0", makeTensorInfo({TensorBoundShape_DimType_CONSTANT}, {16}));
      BoundShapeSpec spec(20, 1000);
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "X0",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 1024});
      verifyShapeInfo(
          out_shape,
          "Out0",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 16});
  */
}


#[test] fn bound_shape_inference_quantization() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "FloatToFused8BitRowwiseQuantized", "", {"w"}, {"Out_w"}, {}));
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "w",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {16, 64}));
      BoundShapeSpec spec(20, 1000);
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "Out_w",
          {TensorBoundShape_DimType_CONSTANT, TensorBoundShape_DimType_CONSTANT},
          {16, 72},
          TensorProto_DataType_UINT8);
  */
}


#[test] fn bound_shape_inference_tile() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Tile",
          "",
          {"blob"},
          {"blob_tile"},
          {MakeArgument<int>("tiles", 32),
           MakeArgument<int>("axis", 0),
           MakeArgument<int>("dynamic", 1)}));
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "blob",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {1, 16}));
      BoundShapeSpec spec(32, 1000);
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "blob_tile",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {32, 16});

      BoundShapeSpec spec2(8, 1000);
      BoundShapeInferencer eng2(spec2);
      eng2.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape2 = eng2.shape_info();
      verifyShapeInfo(
          out_shape2,
          "blob_tile",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {8, 16});
  */
}


#[test] fn bound_shape_inference_combo0() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "SparseLengthsSum", "", {"Weights0", "Data0", "Lengths0"}, {"EB0"}, {}));
      net.add_op()->CopyFrom(CreateOperatorDef(
          "SparseLengthsSum", "", {"Weights1", "Data1", "Lengths1"}, {"EB1"}, {}));
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Concat",
          "",
          {"EB0", "EB1"},
          {"Cout", "split_info"},
          {MakeArgument<int>("axis", 1), MakeArgument<int>("add_axis", 1)}));
      net.add_op()->CopyFrom(CreateOperatorDef(
          "BatchMatMul",
          "",
          {"Cout", "Cout"},
          {"Bout"},
          {MakeArgument<int>("trans_b", 1)}));
      net.add_op()->CopyFrom(
          CreateOperatorDef("Flatten", "", {"Bout"}, {"Fout"}, {}));
      net.add_op()->CopyFrom(
          CreateOperatorDef("BatchGather", "", {"Fout", "Indices"}, {"Gout"}, {}));
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "Weights0",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {1000, 16}));
      shape_map.emplace(
          "Weights1",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {20000, 16}));
      shape_map.emplace(
          "Indices", makeTensorInfo({TensorBoundShape_DimType_CONSTANT}, {2}));
      BoundShapeSpec spec(20, 1000);
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      LOG(INFO) << eng.PrintShapeInfo();
      verifyShapeInfo(
          out_shape,
          "Gout",
          {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
          {spec.max_batch_size, 2});
  */
}


#[test] fn bound_shape_inference_softmax() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Softmax",
          "",
          {"input"},
          {"output"},
          {MakeArgument<int>("axis", 1)}));
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "input",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {1, 16}));
      BoundShapeSpec spec(32, 1000);
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "output",
          {TensorBoundShape_DimType_CONSTANT, TensorBoundShape_DimType_CONSTANT},
          {1, 16});
  */
}


#[test] fn bound_shape_inference_lp_norm() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "LpNorm",
          "",
          {"input"},
          {"output"},
          {MakeArgument<int>("p", 1)}));
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "input",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {1, 16}));
      BoundShapeSpec spec(32, 1000);
      BoundShapeInferencer eng(spec);
      eng.InferBoundShapeAndType(net, shape_map, nullptr);
      const auto& out_shape = eng.shape_info();
      verifyShapeInfo(
          out_shape,
          "output",
          {TensorBoundShape_DimType_CONSTANT},
          {1});
  */
}

