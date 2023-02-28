crate::ix!();

#[test] fn Converter_ClipRangesGatherSigridHashConverter() {
    todo!();
    /*
      OperatorDef op;
      op.set_type("ClipRangesGatherSigridHash");
      op.add_arg()->CopyFrom(caffe2::MakeArgument<bool>("hash_into_int32", true));
      auto nnDef = convertToNeuralNetOperator(op);
      auto* pNNDef =
          static_cast<nom::repr::ClipRangesGatherSigridHash*>(nnDef.get());
      EXPECT_TRUE(pNNDef);
      EXPECT_TRUE(pNNDef->getHashIntoInt32());

      OperatorDef op2;
      op2.set_type("ClipRangesGatherSigridHash");
      op2.add_arg()->CopyFrom(caffe2::MakeArgument<bool>("hash_into_int32", false));
      auto nnDef2 = convertToNeuralNetOperator(op2);
      auto* pNNDef2 =
          static_cast<nom::repr::ClipRangesGatherSigridHash*>(nnDef2.get());
      EXPECT_TRUE(pNNDef2);
      EXPECT_FALSE(pNNDef2->getHashIntoInt32());
  */
}

#[test] fn Converter_ClipRangesGatherSigridHashV2Converter() {
    todo!();
    /*
      OperatorDef op;
      op.set_type("ClipRangesGatherSigridHashV2");
      op.add_arg()->CopyFrom(caffe2::MakeArgument<bool>("hash_into_int32", true));
      auto nnDef = convertToNeuralNetOperator(op);
      auto* pNNDef =
          static_cast<nom::repr::ClipRangesGatherSigridHashV2*>(nnDef.get());
      EXPECT_TRUE(pNNDef);
      EXPECT_TRUE(pNNDef->getHashIntoInt32());

      OperatorDef op2;
      op2.set_type("ClipRangesGatherSigridHashV2");
      op2.add_arg()->CopyFrom(caffe2::MakeArgument<bool>("hash_into_int32", false));
      auto nnDef2 = convertToNeuralNetOperator(op2);
      auto* pNNDef2 =
          static_cast<nom::repr::ClipRangesGatherSigridHashV2*>(nnDef2.get());
      EXPECT_TRUE(pNNDef2);
      EXPECT_FALSE(pNNDef2->getHashIntoInt32());
  */
}
