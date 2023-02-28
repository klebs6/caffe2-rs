crate::ix!();

/**
  | Perform the same operation as Gather,
  | but operating on 8-bit rowwise quantized
  | matrices with fused storage (where
  | each row stores quantized values, and
  | then the scale and offset).
  | 
  | DATA needs to have rank 2 and INDICES
  | needs to have rank 1.
  |
  */
pub struct GatherFused8BitRowwiseOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{GatherFused8BitRowwise, 2}

num_outputs!{GatherFused8BitRowwise, 1}

inputs!{GatherFused8BitRowwise, 
    0 => ("DATA", "uint8 tensor with rank 2 obtained with operator FloatToFused8BitRowwiseQuantized"),
    1 => ("INDICES", "Integer vector containing indices of the first dimension of DATA for the rows that are being gathered")
}

outputs!{GatherFused8BitRowwise, 
    0 => ("OUTPUT", "output")
}

tensor_inference_function!{GatherFused8BitRowwise, /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      for (auto d : in[1].dims()) {
        out[0].add_dims(d);
      }
      for (int i = 1; i < in[0].dims_size(); ++i) {
        out[0].add_dims(in[0].dims(i));
      }
      out[0].set_data_type(in[0].data_type());
      return out;
    } */
}

register_cpu_operator!{
    GatherFused8BitRowwise,
    GatherFused8BitRowwiseOp<CPUContext>
}

input_tags!{
    GatherFused8BitRowwiseOp {
        Data,
        Indices
    }
}

impl<Context> GatherFused8BitRowwiseOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
            this, this->template Input<Tensor>(INDICES, CPU));
        */
    }

    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
        todo!();
        /*
            const auto& data = Input(DATA);
            const auto& indices = Input(INDICES);

            CAFFE_ENFORCE_EQ(data.dim(), 2, "DATA must be a matrix");
            CAFFE_ENFORCE_EQ(indices.dim(), 1, "INDICES must be a vector");
            CAFFE_ENFORCE_GT(data.size(1), 8, "DATA must have more than 8 columns");
            // Subtract 8 from the #columns of data for the 4 bytes for scale and 4
            // bytes for bias that we use in the fused representation (per row).
            const std::vector<int64_t> shape = {indices.size(0), data.size(1) - 8};
            auto* output = Output(0, shape, at::dtype<float>());

            int block_size = shape[1];
            auto block_bytesize = data.size_from_dim(1) * data.dtype().itemsize();
            int N = indices.numel();

            const uint8_t* src_base = data.template data<uint8_t>();
            const Index* idxs = indices.template data<Index>();
            auto out = output->template mutable_data<float>();

            for (int i = 0; i < N; ++i) {
              auto idx = idxs[i];
              CAFFE_ENFORCE(
                  0 <= idx && idx < data.size(0),
                  "INDICES element is out of DATA bounds, id=",
                  idx,
                  " data_dim=",
                  data.size(0));
              const uint8_t* src = src_base + idx * block_bytesize;
              ConstEigenVectorArrayMap<uint8_t> input_row_values(src, shape[1]);
              ConstEigenVectorArrayMap<float> input_row_scale_bias(
                  reinterpret_cast<const float*>(src + shape[1]), 2);

              EigenVectorArrayMap<float> output_row(out + i * shape[1], shape[1]);

              output_row = input_row_values.cast<float>() * input_row_scale_bias(0) +
                  input_row_scale_bias(1);
            }
            return true;
        */
    }
}

