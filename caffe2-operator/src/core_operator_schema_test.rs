crate::ix!();

///Test Documentation
num_inputs!{OpSchemaTestOp, 1}

num_outputs!{OpSchemaTestOp, 1}

inputs!{OpSchemaTestOp, 
    0 => ("in0", "dummy input.")
}

outputs!{OpSchemaTestOp, 
    0 => ("out0", "dummy output.")
}

#[test] fn operator_schema_test_basic_schema() {
    todo!();
    /*
      const OpSchema* schema = OpSchemaRegistry::Schema("OpSchemaTestOp");
    #ifdef CAFFE2_NO_OPERATOR_SCHEMA
      EXPECT_TRUE(schema == nullptr);
      return;
    #endif
      EXPECT_TRUE(schema != nullptr);
      EXPECT_TRUE(schema->doc() != nullptr);
      OperatorDef def1 = CreateOperatorDef(
          "OpSchemaTestOp", "",
          vector<string>{"in"}, vector<string>{"out"});
      EXPECT_TRUE(schema->Verify(def1));
      OperatorDef def2 = CreateOperatorDef(
          "OpSchemaTestOp", "",
          vector<string>{"in1", "in2"}, vector<string>{"out"});
      EXPECT_FALSE(schema->Verify(def2));
      OperatorDef def3 = CreateOperatorDef(
          "OpSchemaTestOp", "",
          vector<string>{"in"}, vector<string>{"out1", "out2"});
      EXPECT_FALSE(schema->Verify(def3));
  */
}

num_inputs!{OpSchemaSpecifiedInputOutputOp, (2,4)}

num_outputs!{OpSchemaSpecifiedInputOutputOp, (1,3)}

#[test] fn operator_schema_test_specified_input_output() {
    todo!();
    /*
      const OpSchema* schema
          = OpSchemaRegistry::Schema("OpSchemaSpecifiedInputOutputOp");
    #ifdef CAFFE2_NO_OPERATOR_SCHEMA
      EXPECT_TRUE(schema == nullptr);
      return;
    #endif
      EXPECT_TRUE(schema != nullptr);
      OperatorDef def1 = CreateOperatorDef(
          "OpSchemaSpecifiedInputOutputOp", "",
          vector<string>{"in"}, vector<string>{"out"});
      EXPECT_FALSE(schema->Verify(def1));
      OperatorDef def2 = CreateOperatorDef(
          "OpSchemaSpecifiedInputOutputOp", "",
          vector<string>{"in1", "in2"}, vector<string>{"out"});
      EXPECT_TRUE(schema->Verify(def2));
      OperatorDef def3 = CreateOperatorDef(
          "OpSchemaSpecifiedInputOutputOp", "",
          vector<string>{"in1", "in2"}, vector<string>{"out1", "out2"});
      EXPECT_FALSE(schema->Verify(def3));
  */
}

num_inputs_outputs!{OpSchemaInputOutputRelationOp, 
    |input: i32, output: i32| {
        output == input || output == input * 2
    }
}

#[test] fn operator_schema_test_input_output_relation() {
    todo!();
    /*
      const OpSchema* schema
          = OpSchemaRegistry::Schema("OpSchemaInputOutputRelationOp");
    #ifdef CAFFE2_NO_OPERATOR_SCHEMA
      EXPECT_TRUE(schema == nullptr);
      return;
    #endif
      EXPECT_TRUE(schema != nullptr);
      OperatorDef def1 = CreateOperatorDef(
          "OpSchemaInputOutputRelationOp", "",
          vector<string>{"in"}, vector<string>{"out"});
      EXPECT_TRUE(schema->Verify(def1));
      OperatorDef def2 = CreateOperatorDef(
          "OpSchemaInputOutputRelationOp", "",
          vector<string>{"in"}, vector<string>{"out1", "out2"});
      EXPECT_TRUE(schema->Verify(def2));
      OperatorDef def3 = CreateOperatorDef(
          "OpSchemaInputOutputRelationOp", "",
          vector<string>{"in1", "in2", "in3"}, vector<string>{"out1", "out2"});
      EXPECT_FALSE(schema->Verify(def3));
  */
}

same_number_of_output!{OpSchemaSameInputOutputOp}

#[test] fn operator_schema_test_same_input_output() {
    todo!();
    /*
      const OpSchema* schema =
          OpSchemaRegistry::Schema("OpSchemaSameInputOutputOp");
    #ifdef CAFFE2_NO_OPERATOR_SCHEMA
      EXPECT_TRUE(schema == nullptr);
      return;
    #endif
      OperatorDef def1 = CreateOperatorDef(
          "OpSchemaSameInputOutputOp", "",
          vector<string>{"in"}, vector<string>{"out"});
      EXPECT_TRUE(schema->Verify(def1));
      OperatorDef def2 = CreateOperatorDef(
          "OpSchemaSameInputOutputOp", "",
          vector<string>{"in1", "in2"}, vector<string>{"out1", "out2"});
      EXPECT_TRUE(schema->Verify(def2));
      OperatorDef def3 = CreateOperatorDef(
          "OpSchemaSameInputOutputOp", "",
          vector<string>{"in1", "in2"}, vector<string>{"out1", "out2", "out3"});
      EXPECT_FALSE(schema->Verify(def3));
  */
}

num_inputs!{OpSchemaCalculateOutputOp, (1,5)}

num_outputs!{OpSchemaCalculateOutputOp, (2,6)}

output_calculator!{OpSchemaCalculateOutputOp, /*[](int n) { return n + 1; }*/}

#[test] fn operator_schema_test_calculate_output() {
    todo!();
    /*
      const OpSchema* schema =
          OpSchemaRegistry::Schema("OpSchemaCalculateOutputOp");
    #ifdef CAFFE2_NO_OPERATOR_SCHEMA
      EXPECT_TRUE(schema == nullptr);
      return;
    #endif
      OperatorDef def1 = CreateOperatorDef(
          "OpSchemaCalculateOutputOp", "",
          vector<string>{"in"}, vector<string>{"out"});
      EXPECT_FALSE(schema->Verify(def1));
      OperatorDef def2 = CreateOperatorDef(
          "OpSchemaCalculateOutputOp", "",
          vector<string>{"in1", "in2"}, vector<string>{"out1", "out2"});
      EXPECT_FALSE(schema->Verify(def2));
      OperatorDef def3 = CreateOperatorDef(
          "OpSchemaCalculateOutputOp", "",
          vector<string>{"in1", "in2"}, vector<string>{"out1", "out2", "out3"});
      EXPECT_TRUE(schema->Verify(def3));
  */
}

num_inputs!{OpSchemaInplace, 2}

num_outputs!{OpSchemaInplace, 2}

allow_inplace!{OpSchemaInplace, vec![(0, 0)]}

enforce_inplace!{OpSchemaInplace, vec![(1, 1)]}

#[test] fn operator_schema_test_inplace() {
    todo!();
    /*
      const OpSchema* schema =
          OpSchemaRegistry::Schema("OpSchemaInplace");
    #ifdef CAFFE2_NO_OPERATOR_SCHEMA
      EXPECT_TRUE(schema == nullptr);
      return;
    #endif
      OperatorDef def1 = CreateOperatorDef(
          "OpSchemaInplace", "",
          vector<string>{"in1", "in2"}, vector<string>{"out1", "in2"});
      EXPECT_TRUE(schema->Verify(def1));
      OperatorDef def2 = CreateOperatorDef(
          "OpSchemaInplace", "",
          vector<string>{"in1", "in2"}, vector<string>{"in1", "in2"});
      EXPECT_TRUE(schema->Verify(def2));
      OperatorDef def3 = CreateOperatorDef(
          "OpSchemaInplace", "",
          vector<string>{"in1", "in2"}, vector<string>{"in1", "out2"});
      EXPECT_FALSE(schema->Verify(def3));
      OperatorDef def4 = CreateOperatorDef(
          "OpSchemaInplace", "",
          vector<string>{"in1", "in2"}, vector<string>{"out1", "out2"});
      EXPECT_FALSE(schema->Verify(def4));
  */
}

identical_type_and_shape!{OpSchemaSameInputOutputTensorInference}

#[test] fn operator_schema_test_tensor_inference_identical() {
    todo!();
    /*
      const OpSchema* schema =
          OpSchemaRegistry::Schema("OpSchemaSameInputOutputTensorInference");
    #ifdef CAFFE2_NO_OPERATOR_SCHEMA
      EXPECT_TRUE(schema == nullptr);
      return;
    #endif
      OperatorDef def = CreateOperatorDef(
          "OpSchemaSameInputOutputTensorInference",
          "",
          vector<string>{"in"},
          vector<string>{"out"});
      vector<TensorShape> shapes(1);
      shapes[0].set_data_type(TensorProto::FLOAT);
      shapes[0].add_dims(1);
      shapes[0].add_dims(2);
      shapes[0].add_dims(3);
      vector<TensorShape> out = schema->InferTensor(def, shapes);
      EXPECT_EQ(out.size(), 1);
      EXPECT_EQ(out[0].SerializeAsString(), shapes[0].SerializeAsString());
  */
}

tensor_inference_function!{OpSchemaArbitraryTensorInference, /* (
        [](const OperatorDef&, const vector<TensorShape>&) {
          vector<TensorShape> shapes(1);
          shapes[0].set_data_type(TensorProto::FLOAT);
          shapes[0].add_dims(1701);
          return shapes;
        }) */
}

#[test] fn operator_schema_test_tensor_inference_arbitrary() {
    todo!();
    /*
      const OpSchema* schema =
          OpSchemaRegistry::Schema("OpSchemaArbitraryTensorInference");
    #ifdef CAFFE2_NO_OPERATOR_SCHEMA
      EXPECT_TRUE(schema == nullptr);
      return;
    #endif
      OperatorDef def = CreateOperatorDef(
          "OpSchemaArbitraryTensorInference",
          "",
          vector<string>{"in"},
          vector<string>{"out"});
      vector<TensorShape> out = schema->InferTensor(def, vector<TensorShape>());
      EXPECT_EQ(out.size(), 1);
      EXPECT_EQ(out[0].data_type(), TensorProto::FLOAT);
      EXPECT_EQ(out[0].dims_size(), 1);
      EXPECT_EQ(out[0].dims(0), 1701);
  */
}


#[test] fn operator_schema_test_cast_schema() {
    todo!();
    /*
      // This tests a use case of the schema: the Cast op takes in the def and
      // deduces the
      // schema from the "to" argument.
      const OpSchema* schema = OpSchemaRegistry::Schema("Cast");
    #ifdef CAFFE2_NO_OPERATOR_SCHEMA
      EXPECT_TRUE(schema == nullptr);
      return;
    #endif
      if (!schema) {
        // Compiled without the Cast op.
        return;
      }
      OperatorDef def = CreateOperatorDef(
          "Cast",
          "",
          vector<string>{"in"},
          vector<string>{"out"},
          vector<Argument>{MakeArgument<int64_t>("to", TensorProto::UINT8)});
      vector<TensorShape> out = schema->InferTensor(def, vector<TensorShape>(1));
      EXPECT_EQ(out.size(), 1);
      // Data type should be inferred.
      EXPECT_EQ(out[0].data_type(), TensorProto::UINT8);
      // Dim should not be set (same as input);
      EXPECT_EQ(out[0].dims_size(), 0);
  */
}

num_inputs!{OpSchemaCostInference, 2}

num_outputs!{OpSchemaCostInference, 2}

cost_inference_function!{OpSchemaCostInference, 
    /* ([](const OperatorDef& /*def*/,
                              const vector<TensorShape>& inputs) {
      struct OpSchema::Cost c;
      c.flops = 2 * inputs[0].dims(0) * inputs[0].dims(1) * inputs[1].dims(1);
      return c;
    }) */ 
}


#[test] fn operator_schema_test_cost_inference() {
    todo!();
    /*
      const OpSchema* schema = OpSchemaRegistry::Schema("OpSchemaCostInference");
    #ifdef CAFFE2_NO_OPERATOR_SCHEMA
      EXPECT_TRUE(schema == nullptr);
      return;
    #endif
      if (!schema) {
        return;
      }
      OperatorDef def = CreateOperatorDef(
          "OpSchemaCostInference", "", vector<string>{"in"}, vector<string>{"out"});
      vector<TensorShape> shapes(2);
      shapes[0].set_data_type(TensorProto::FLOAT);
      shapes[0].add_dims(10);
      shapes[0].add_dims(10);
      shapes[1].set_data_type(TensorProto::FLOAT);
      shapes[1].add_dims(10);
      shapes[1].add_dims(10);
      EXPECT_EQ(2000, schema->InferCost(def, shapes).flops);
  */
}

