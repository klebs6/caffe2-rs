crate::ix!();

#[test] fn in_batch_broadcast_main() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(
          CreateOperatorDef("FloatToHalf", "", {"blob"}, {"blob_half"}, {}));
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "blob",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
              {32, 16}));
      std::unordered_set<std::string> transform_blob({"blob"});
      opt::inBatchBroadcast(&net, transform_blob, 32, shape_map);
      NetDef expected_net;
      auto* op1 = expected_net.add_op();
      op1->CopyFrom(CreateOperatorDef(
          "FloatToHalf",
          "",
          {"blob"},
          {"blob_fp16"},
          {}));
      op1->mutable_device_option()->set_device_type(caffe2::PROTO_CPU);
      auto* op2 = expected_net.add_op();
      op2->CopyFrom(CreateOperatorDef(
          "HalfToFloat",
          "",
          {"blob_fp16"},
          {"blob_fp32"},
          {}));
      op2->mutable_device_option()->set_device_type(caffe2::PROTO_CPU);
      auto op3 = expected_net.add_op();
      op3->CopyFrom(CreateOperatorDef(
          "Tile",
          "",
          {"blob_fp32"},
          {"blob_tile"},
          {MakeArgument<int>("tiles", 32),
           MakeArgument<int>("axis", 0),
           MakeArgument<int>("dynamic", 1)}));
      op3->mutable_device_option()->set_device_type(caffe2::PROTO_CPU);
      auto op4 = expected_net.add_op();
      op4->CopyFrom(
          CreateOperatorDef("FloatToHalf", "", {"blob_tile"}, {"blob_half"}, {}));
      op4->mutable_device_option()->set_device_type(caffe2::PROTO_CPU);
      ShapeInfoMap expected_shape_map;
      expected_shape_map.emplace(
          "blob",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {1, 16}));
      expected_shape_map.emplace(
          "blob_fp16",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {1, 16},
              TensorProto_DataType_FLOAT16));
      expected_shape_map.emplace(
          "blob_fp32",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {1, 16}));
      expected_shape_map.emplace(
          "blob_tile",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
              {32, 16}));
      checkNet(net, expected_net);
      checkShapeInfo(shape_map, expected_shape_map);
      */
}

#[test] fn in_batch_broadcast_fuse8bit() {
    todo!();
    /*
      NetDef net;
      net.add_op()->CopyFrom(CreateOperatorDef(
          "Fused8BitRowwiseQuantizedToFloat", "", {"blob_int8"}, {"blob"}, {}));
      ShapeInfoMap shape_map;
      shape_map.emplace(
          "blob_int8",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
              {32, 24},
              TensorProto_DataType_UINT8));
      shape_map.emplace(
          "blob",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
              {32, 16}));
      std::unordered_set<std::string> transform_blob({"blob_int8"});
      opt::inBatchBroadcast(&net, transform_blob, 32, shape_map);
      NetDef expected_net;
      auto* op1 = expected_net.add_op();
      op1->CopyFrom(CreateOperatorDef(
          "Fused8BitRowwiseQuantizedToFloat", "", {"blob_int8"}, {"blob"}, {}));
      op1->mutable_device_option()->set_device_type(caffe2::PROTO_CPU);
      auto* op2 = expected_net.add_op();
      op2->CopyFrom(CreateOperatorDef(
          "FloatToHalf",
          "",
          {"blob"},
          {"blob_fp16"},
          {}));
      op2->mutable_device_option()->set_device_type(caffe2::PROTO_CPU);
      auto* op3 = expected_net.add_op();
      op3->CopyFrom(CreateOperatorDef(
          "HalfToFloat",
          "",
          {"blob_fp16"},
          {"blob_fp32"},
          {}));
      op3->mutable_device_option()->set_device_type(caffe2::PROTO_CPU);
      auto* op4 = expected_net.add_op();
      op4->CopyFrom(CreateOperatorDef(
          "Tile",
          "",
          {"blob_fp32"},
          {"blob_tile"},
          {MakeArgument<int>("tiles", 32),
           MakeArgument<int>("axis", 0),
           MakeArgument<int>("dynamic", 1)}));
      op4->mutable_device_option()->set_device_type(caffe2::PROTO_CPU);
      ShapeInfoMap expected_shape_map;
      expected_shape_map.emplace(
          "blob_int8",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {1, 24},
              TensorProto_DataType_UINT8));
      expected_shape_map.emplace(
          "blob",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {1, 16}));
      expected_shape_map.emplace(
          "blob_fp16",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {1, 16},
              TensorProto_DataType_FLOAT16));
      expected_shape_map.emplace(
          "blob_fp32",
          makeTensorInfo(
              {TensorBoundShape_DimType_CONSTANT,
               TensorBoundShape_DimType_CONSTANT},
              {1, 16},
              TensorProto_DataType_FLOAT));
      expected_shape_map.emplace(
          "blob_tile",
          makeTensorInfo(
              {TensorBoundShape_DimType_BATCH, TensorBoundShape_DimType_CONSTANT},
              {32, 16}));
      checkNet(net, expected_net);
      checkShapeInfo(shape_map, expected_shape_map);
      */
}
