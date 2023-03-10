crate::ix!();

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
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct CastOp<Context> {
    storage: OperatorStorage,
    context: Context,
    body:    fn() -> bool,
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
    }*/
}
