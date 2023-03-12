crate::ix!();

#[macro_export] macro_rules! caffe2_schema_for_binary_comparison_op {
    ($name:ident, $symbol:expr, $desc:expr, $extra:ident) => {
        /*
        
          OPERATOR_SCHEMA(name)                                                     
              .NumInputs(2)                                                         
              .NumOutputs(1)                                                        
              .TensorInferenceFunction([](const OperatorDef& def,                   
                                          const vector<TensorShape>& in) {          
                ArgumentHelper helper(def);                                         
                const auto broadcasted =                                            
                    helper.GetSingleArgument<bool>("broadcast", false);             
                if (!broadcasted) {                                                 
                  CAFFE_ENFORCE_EQ(in[0].dims().size(), in[1].dims().size());       
                  for (int i = 0; i < in[0].dims().size(); ++i) {                   
                    CAFFE_ENFORCE_EQ(in[0].dims(i), in[1].dims(i));                 
                  }                                                                 
                }                                                                   
                auto output_dims =                                                  
                    std::vector<int64_t>(in[0].dims().begin(), in[0].dims().end()); 
                return vector<TensorShape>{                                         
                    CreateTensorShape(output_dims, TensorProto::BOOL)};             
              })                                                                    
              .FillUsing(ComparisonDocGenerator(symbol, desc, extra));              
          SHOULD_NOT_DO_GRADIENT(name)
        */
    }
}

caffe2_schema_for_binary_comparison_op!{EQ, "==",  "equal to",                kEQExample}
caffe2_schema_for_binary_comparison_op!{NE, "!=",  "not equal to",            kNEExample}
caffe2_schema_for_binary_comparison_op!{LT, "<",   "less than",               kLTExample}
caffe2_schema_for_binary_comparison_op!{LE, "<=",  "less or equal than",      kLEExample}
caffe2_schema_for_binary_comparison_op!{GT, ">",   "greater than",            kGTExample}
caffe2_schema_for_binary_comparison_op!{GE, ">=",  "greater or equal than",   kGEExample}

#[macro_export] macro_rules! caffe2_schema_for_binary_logical_op {
    ($name:ident, $symbol:expr, $onnx_schema:expr, $extra:ident) => {
        /*
        
          OPERATOR_SCHEMA(name)                                                       
              .NumInputs(2)                                                           
              .NumOutputs(1)                                                          
              .AllowInplace({{0, 0}})                                                 
              .FillUsing(LogicalDocGenerator(symbol, extra))                          
              .TensorInferenceFunction(ElementwiseOpShapeInference)                   
              .InheritOnnxSchema(onnx_schema);                                        
          SHOULD_NOT_DO_GRADIENT(name)
        */
    }
}

caffe2_schema_for_binary_logical_op!{Or, "or", "Or", kOrExample}
caffe2_schema_for_binary_logical_op!{And, "and", "And", kAndExample}
caffe2_schema_for_binary_logical_op!{Xor, "xor", "Xor", kXorExample}

#[macro_export] macro_rules! caffe2_schema_for_binary_bitwise_op {
    ($name:ident, $symbol:expr) => {
        /*
        
          OPERATOR_SCHEMA(name)                                      
              .NumInputs(2)                                          
              .NumOutputs(1)                                         
              .AllowInplace({{0, 0}})                                
              .FillUsing(BitwiseDocGenerator(symbol))                
              .TensorInferenceFunction(ElementwiseOpShapeInference); 
          SHOULD_NOT_DO_GRADIENT(name)
        */
    }
}

caffe2_schema_for_binary_bitwise_op!{BitwiseOr, "bitwise_or"}
caffe2_schema_for_binary_bitwise_op!{BitwiseAnd, "bitwise_and"}
caffe2_schema_for_binary_bitwise_op!{BitwiseXor, "bitwise_xor"}
