crate::ix!();

use crate::{
    OpSchema,
    OperatorDef,
    TensorShape
};

pub const kBroadcastDoc: &'static str = "
| If necessary the right-hand-side argument will be
| broadcasted to match the shape of
| left-hand-side argument. When broadcasting is
| specified, the second tensor can either be of
| size 1 (a scalar value), or having its shape
| as a contiguous subset of the first tensor's
| shape. The starting of the mutually equal
| shape is specified by the argument \"axis\",
| and if it is not set, suffix matching is
| assumed. 1-dim expansion doesn't work yet.
|
| For example, the following tensor shapes are
| supported (with broadcast=1):
|
| ```
|   shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
|   shape(A) = (2, 3, 4, 5), shape(B) = (5,)
|   shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
|   shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
|   shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
| ```
| Argument `broadcast=1` needs to be passed to
| enable broadcasting.
|
| Github Links:
|
| - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/elementwise_ops_schema.cc
";

    
pub fn math_doc_generator(
    name: &str, 
    extra: &str) -> fn(schema: &OpSchema) -> () 
{
    todo!();
    /*
      return [=](OpSchema& schema) {
        string doc = R"DOC(
    Performs element-wise binary {name} (with limited broadcast support).
    {broadcast_doc}

    {extra}
    )DOC";
        c10::ReplaceAll(doc, "{name}", name);
        c10::ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
        c10::ReplaceAll(doc, "{extra}", extra);
        schema.SetDoc(doc);
        schema.Arg(
            "broadcast", "*(type: int; default: 0)* Pass 1 to enable broadcasting");
        schema.Arg("axis", "*(type: int; default: -1)* Axis to concatenate on.");
        schema.Input(
            0,
            "A",
            "*(type: Tensor`<float>`)* First operand, should share the type with the second operand.");
        schema.Input(
            1,
            "B",
            "*(type: Tensor`<float>`)* Second operand. With broadcasting can be of smaller size than A. "
            "If broadcasting is disabled it should be of the same size as A.");
        schema.Output(
            0,
            "C",
            "*(type: Tensor`<float>`)* Output tensor with same dimensions and type as A.");
      };

    */
}

#[inline] pub fn elementwise_op_shape_inference(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        std::vector<TensorShape> out(1);
      out[0].set_data_type(in[0].data_type());
      ArgumentHelper helper(def);
      const bool broadcast = helper.GetSingleArgument<bool>("broadcast", false);
      if (broadcast) {
        out[0].mutable_dims()->CopyFrom(in[0].dims());
      } else {
        const std::vector<int> A_dims(in[0].dims().begin(), in[0].dims().end());
        const std::vector<int> B_dims(in[1].dims().begin(), in[1].dims().end());
        const std::vector<int> C_dims =
            elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
                A_dims, B_dims);
        for (const int dim : C_dims) {
          out[0].add_dims(dim);
        }
      }
      return out;
    */
}


#[inline] pub fn elementwise_gradient_op_shape_inference(
    def:   &OperatorDef,
    input: &Vec<TensorShape>) -> Vec<TensorShape> 
{
    todo!();
    /*
        std::vector<TensorShape> out;
      out.push_back(in.at(1));
      out.push_back(in.at(2));
      return out;
    */
}

#[inline] pub fn comparison_doc_generator(
    name:  *const u8,
    desc:  *const u8,
    extra: *const u8) -> fn(_u0: &mut OpSchema) -> c_void {
    
    todo!();
    /*
        return [=](OpSchema& schema) {
        string doc = R"DOC(
    Performs element-wise {desc} comparison **{name}** (with limited broadcast support).

    {broadcast_doc}

    {extra}
    )DOC";
        c10::ReplaceAll(doc, "{name}", name);
        c10::ReplaceAll(doc, "{desc}", desc);
        c10::ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
        c10::ReplaceAll(doc, "{extra}", extra);
        schema.SetDoc(doc);
        schema.Arg( "broadcast", "*(type: int; default: 0)* Pass 1 to enable broadcasting.");
        schema.Arg( "axis", "*(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.");
        schema.Input( 0, "A", "*(type: Tensor`<bool>`)* First operand, should share the type with the second operand.");
        schema.Input( 1, "B", "*(type: Tensor`<bool>`)* Second operand. With broadcasting can be of smaller size than `A`. If broadcasting is disabled it should be of the same size.");
        schema.Output( 0, "C", "*(type: Tensor`<bool>`)* Output tensor with same dimensions as `A`.");
      };
    */
}

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

#[inline] pub fn logical_doc_generator(
    name: *const u8, 
    extra: *const u8) -> fn(_u0: &mut OpSchema) -> c_void 
{
    todo!();
    /*
        return [=](OpSchema& schema) {
        string doc = R"DOC(
    Performs element-wise logical operation **{name}** (with limited broadcast support).
    Both input operands should be of type `bool`.

    {broadcast_doc}

    {extra}
        )DOC";
        c10::ReplaceAll(doc, "{name}", name);
        c10::ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
        c10::ReplaceAll(doc, "{extra}", extra);
        schema.SetDoc(doc);
        schema.Arg(
            "broadcast",
            "*(type: int; default: 0)* Pass 1 to enable broadcasting.");
        schema.Arg(
            "axis",
            "*(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.");
        schema.Input(0, "A", "*(type: Tensor`<bool>`)* First operand.");
        schema.Input(
            1,
            "B",
            "*(type: Tensor`<bool>`)* Second operand. With broadcasting can be of smaller size than `A`. "
            "If broadcasting is disabled it should be of the same size.");
        schema.Output(
            0,
            "C",
            "*(type: Tensor`<bool>`)* Output tensor of booleans. Has same dimensions as input `A`.");
      };
    */
}


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

#[inline] pub fn bitwise_doc_generator(
    name: *const u8) -> fn(_u0: &mut OpSchema) -> c_void 
{
    todo!();
    /*
        return [=](OpSchema& schema) {
        string doc = R"DOC(
    Performs element-wise bitwise operation `{name}` (with limited broadcast support).
    Both input operands should be of type `bool`.
    {broadcast_doc})DOC";
        c10::ReplaceAll(doc, "{name}", name);
        c10::ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
        schema.SetDoc(doc);
        schema.Arg(
            "broadcast",
            "*(type: int; default: 0)* Pass 1 to enable broadcasting.");
        schema.Arg(
            "axis",
            "*(type: int; default: -1)* Axis to concatenate on. If set, defines the broadcast dimensions.");
        schema.Input(0, "A", "*(type: Tensor)* First operand.");
        schema.Input(
            1,
            "B",
            "*(type: Tensor)* Second operand. With broadcasting can be of smaller size than `A`. "
            "If broadcasting is disabled it should be of the same size.");
        schema.Output(
            0,
            "C",
            "*(type: Tensor)* Output tensor. Has same dimensions as input `A`.");
      };
    */
}

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
