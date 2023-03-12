crate::ix!();

/**
  | Applies 8-bit row-wise quantization
  | by determining the range (maximum -
  | minimum) and offset (minimum value)
  | of each row in the input matrix, and then
  | scaling each element to an 8-bit number
  | between 0 and 255.
  | 
  | To later de-quantize values, the scale
  | (range / 255) and offset (bias) are stored
  | alongside the data.
  | 
  | More precisely, each row contains int8
  | elements for each quantized element,
  | and the last 8 bytes of each row in the
  | output matrix are a float storing the
  | scale followed by another float containing
  | the scale.
  | 
  | For N-dimensional input tensor, the
  | first N-1 dimensions are interpreted
  | as rows and the last dimension is interpreted
  | as a column.
  | 
  | For example, an input tensor with dimension
  | 5x2x4 is interpreted as 10 rows and 4
  | columns.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FloatToFused8BitRowwiseQuantizedOp<T, TypeForScaleAndBias, ConvertFn, const HAS_CONVERT: bool, Context> {
    storage:                    OperatorStorage,
    context:                    Context,
    phantom:                    PhantomData<T>,
    phantomCFN:                 PhantomData<ConvertFn>,
    phantomTypeForScaleAndBias: PhantomData<TypeForScaleAndBias>,
}

num_inputs!{FloatToFused8BitRowwiseQuantized, 1}

num_outputs!{FloatToFused8BitRowwiseQuantized, 1}

inputs!{FloatToFused8BitRowwiseQuantized, 
    0 => ("input", "Float32 input data")
}

outputs!{FloatToFused8BitRowwiseQuantized, 
    0 => ("output", "Fused scale, bias and quantized data")
}

tensor_inference_function!{FloatToFused8BitRowwiseQuantized, /* [](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      X.set_dims(
          X.dims().size() - 1, X.dims(X.dims().size() - 1) + 2 * sizeof(float));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_UINT8);
      return out;
    } */
}

no_gradient!{FloatToFused8BitRowwiseQuantized}

input_tags!{
    FloatToFused8BitRowwiseQuantizedOp {
        DataFloat
    }
}

output_tags!{
    FloatToFused8BitRowwiseQuantizedOp {
        DataFusedScaleBiasInt8
    }
}
