crate::ix!();

use crate::{
    OperatorDef,
    Workspace,
    TensorProto_DataType,
    OperatorStorage,
    CPUContext,
    GradientMakerBase
};

#[test] fn cast_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Cast",
        ["X"],
        ["Y"],
        to=2
    )

    workspace.FeedBlob("X", (np.random.rand(3,3)).astype(np.float32)*10)
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X: [[9.436466   5.8529844  0.54932857]
     [1.1583444  2.9936118  0.22950427]
     [3.9143739  3.4040766  8.905341  ]]
    Y: [[9 5 0]
     [1 2 0]
     [3 3 8]]
    */
}

/**
| Casts the elements of a given input tensor to
| a data type specified by the `to` argument and
| returns an output tensor of the same size in the
| converted type.
|
| The `to` argument must be one of the data types
| specified in the *DataType* enum field in the
| TensorProto message (see below). If the `to`
| argument is not provided or is not one of the
| enumerated types in *DataType*, Caffe2 throws an
| Enforce error.
|
| NOTE: Casting from strings is not supported, and
| casting to strings is only supported on CPU.
|
| TensorProto *DataType* field:
| ```
| message TensorProto {
|   ...
|   enum DataType {
|     UNDEFINED = 0;
|     FLOAT     = 1;   // float
|     INT32     = 2;   // int
|     BYTE      = 3;   // BYTE, when deserialized, is going to be restored as uint8.
|     STRING    = 4;   // string
|     BOOL      = 5;   // bool
|     UINT8     = 6;   // uint8_t
|     INT8      = 7;   // int8_t
|     UINT16    = 8;   // uint16_t
|     INT16     = 9;   // int16_t
|     INT64     = 10;  // int64_t
|     FLOAT16   = 12;  // at::Half
|     DOUBLE    = 13;  // double
|   }
| ```
|
| Github Links:
|
| - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/cast_op.cc
|
*/
pub struct CastOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    body: fn() -> bool,
}

impl<Context> CastOp<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws) 

        const ArgumentHelper helper(operator_def);
        TensorProto_DataType to = cast::GetCastDataType(helper, "to");
        TensorProto_DataType from = cast::GetCastDataType(helper, "from_type");

        SetBody(to);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return (this->*body_)();
        */
    }

    #[inline] pub fn do_run_with_type<DstType, SrcType>(&mut self) -> bool {
        todo!();
        /*
            auto& input = Input(0);
            auto* output = Output(0);
            output->ResizeLike(input);
            const auto* data = input.template data<SrcType>();
            auto* out = output->template mutable_data<DstType>();
            auto N = input.size();
            for (int64_t i = 0; i < N; ++i) {
              out[i] = static_cast<DstType>(data[i]);
            }
            return true;
        */
    }
}

impl CastOp<CPUContext> {

    #[inline] pub fn do_run_with_dst_type<DstType>(&mut self) -> bool {
        todo!();
        /*
            return DispatchHelper<
              TensorTypes<
                  float,
                  int32_t,
                  bool,
                  uint8_t,
                  int8_t,
                  uint16_t,
                  int16_t,
                  int64_t,
                  double>,
              DstType>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_cpu_with_type<DstType, SrcType>(&mut self) -> bool {
        todo!();
        /*
            auto& input = Input(0);

          auto* output = Output(0, input.sizes(), at::dtype<DstType>());
          const auto* data = input.template data<SrcType>();
          auto* out = output->template mutable_data<DstType>();
          auto N = input.numel();
          for (int64_t i = 0; i < N; ++i) {
            out[i] = CastHelper<DstType, SrcType>::call(data[i]);
          }
          return true;
        */
    }
    
    /// Allow for Context-specific implementations
    #[inline] pub fn set_body(&mut self, to: TensorProto_DataType)  {
        
        todo!();
        /*
            switch (to) {
        case TensorProto_DataType_FLOAT:
          // body_ = &CastOp::DoRunIncFp16WithDstType<float>;
          body_ = &CastOp<CPUContext>::DoRunWithDstType<float>;
          break;
        case TensorProto_DataType_INT32:
          body_ = &CastOp<CPUContext>::DoRunWithDstType<int>;
          break;
        case TensorProto_DataType_BYTE:
          LOG(FATAL) << "BYTE is deprecated";
          break;
        case TensorProto_DataType_STRING:
          body_ = &CastOp<CPUContext>::DoRunWithDstType<std::string>;
          break;
        case TensorProto_DataType_BOOL:
          body_ = &CastOp<CPUContext>::DoRunWithDstType<bool>;
          break;
        case TensorProto_DataType_UINT8:
          body_ = &CastOp<CPUContext>::DoRunWithDstType<uint8_t>;
          break;
        case TensorProto_DataType_INT8:
          body_ = &CastOp<CPUContext>::DoRunWithDstType<int8_t>;
          break;
        case TensorProto_DataType_UINT16:
          body_ = &CastOp<CPUContext>::DoRunWithDstType<uint16_t>;
          break;
        case TensorProto_DataType_INT16:
          body_ = &CastOp<CPUContext>::DoRunWithDstType<int16_t>;
          break;
        case TensorProto_DataType_INT64:
          body_ = &CastOp<CPUContext>::DoRunWithDstType<int64_t>;
          break;
        case TensorProto_DataType_FLOAT16:
          CAFFE_THROW("Casting to and from at::Half on CPU is not supported yet");
          // break;
        case TensorProto_DataType_DOUBLE:
          // body_ = &CastOp::DoRunIncFp16WithDstType<double>;
          body_ = &CastOp<CPUContext>::DoRunWithDstType<double>;
          break;
        case TensorProto_DataType_UNDEFINED:
          CAFFE_THROW("Cast op must have 'to' argument of type DataType");
          // break;
        default:
          CAFFE_THROW("Unexpected 'to' argument value: ", to);
      }
        */
    }
}

pub struct CastHelper<DstType, SrcType> {
    phantomA: PhantomData<DstType>,
    phantomB: PhantomData<SrcType>,
}

impl<DstType,SrcType> CastHelper<DstType, SrcType> {
    
    #[inline] pub fn call(data: SrcType) -> DstType {
        
        todo!();
        /*
            return static_cast<DstType>(data);
        */
    }
}

register_cpu_operator!{Cast, CastOp<CPUContext>}

num_inputs!{Cast, 1}

num_outputs!{Cast, 1}

inputs!{Cast, 
    0 => ("X", "*(type: Tensor)* Input tensor to be cast.")
}

outputs!{Cast, 
    0 => ("Y", "*(type: Tensor`<'to' type>`)* Output tensor with the same shape as input with type specified by the `to` argument.")
}

args!{Cast, 
    0 => ("to", "*(type: int)* Data type to which the elements of the input tensor are cast. Strictly must be one of the types from *DataType* enum in TensorProto.")
}

inherit_onnx_schema!{Cast}

tensor_inference_function!{Cast, /*[](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      vector<TensorShape> out;
      out.push_back(in[0]);
      out[0].set_data_type(cast::GetCastDataType(helper, "to"));
      return out;
    }*/}

/**
  | Some Casts are compatible with gradients,
  | but for now we don't support it
  | 
  | GRADIENT_NOT_IMPLEMENTED_YET(Cast);
  |
  */
pub struct GetCastGradient;

impl GetGradientDefs for GetCastGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            vector<OperatorDef> defs = SingleGradientDef("Cast", "", vector<string>{GO(0)}, vector<string>{GI(0)});

        // now modify the arguments in defs[0]
        ArgumentHelper argsHelper(def_);

        auto to_name = cast::GetCastDataType(argsHelper, "to");

        CAFFE_ENFORCE(
            argsHelper.HasSingleArgumentOfType<string>("from_type") ||
                argsHelper.HasSingleArgumentOfType<int>("from_type"),
            "Argument 'from_type' of type int or string"
            " is required to get the gradient of CastOp");

        auto from_name = cast::GetCastDataType(argsHelper, "from_type");
        Argument *to = defs[0].add_arg();
        to->set_name("to");
        to->set_i(from_name);

        Argument *from = defs[0].add_arg();
        from->set_name("from_type");
        from->set_i(to_name);

        return defs;
        */
    }
}

impl CopyArguments for GetCastGradient {
    
    #[inline] fn copy_arguments(&self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}

register_gradient!{Cast, GetCastGradient}
