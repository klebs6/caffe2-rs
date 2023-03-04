crate::ix!();

/**
  | Produces a slice of the input Int8 tensor.
  | 
  | Currently, only slicing in a single
  | dimension is supported.
  | 
  | Slices are passed as 2 1D vectors or as
  | two keyword argument lists with starting
  | and end indices for each dimension of
  | the input `data` tensor.
  | 
  | If a negative value is passed for any
  | of the start or end indices, it represents
  | the number of elements before the end
  | of that dimension.
  | 
  | End indices are non-inclusive unless
  | negative (end index -1 means up to and
  | including the last element).
  |
  */
pub struct Int8SliceOp {
    base: SliceOp<CPUContext>,
}

register_cpu_operator!{Int8Slice, int8::Int8SliceOp}

num_inputs!{Int8Slice, (1,3)}

num_outputs!{Int8Slice, 1}

inputs!{Int8Slice, 
    0 => ("data",           "Int8 Tensor of data to extract slices from."),
    1 => ("starts",         "1D tensor: start-indices for each dimension of data."),
    2 => ("ends",           "1D tensor: end-indices for each dimension of data.")
}

outputs!{Int8Slice, 
    0 => ("output",         "Sliced Int8 data tensor.")
}

args!{Int8Slice, 
    0 => ("Y_scale",        "Output tensor quantization scale"),
    1 => ("Y_zero_point",   "Output tensor quantization offset"),
    2 => ("starts",         "List of starting indices"),
    3 => ("ends",           "List of ending indices"),
    4 => ("dim",            "(Optional) The dimension to slice over. If specified start_idx and end_idx should also be given and it takes precedence over starts and ends"),
    5 => ("start_idx",      "(Optional) The dimension to start slice from. Default is 0"),
    6 => ("end_idx",        "(Optional) The dimension to end the slice. Default is -1")
}

inherit_onnx_schema!{Int8Slice, "Slice"}

#[test] fn int8_slice_op_example() {

    todo!();

    /*
    Example:

      data = [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
      ]
      starts = [0, 1]
      ends = [-1, 3]

      result = [
          [2, 3],
          [6, 7],
      ]
    */
}

impl Int8SliceOp {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : SliceOp(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (InputSize() > 1) {
          return DispatchHelper<TensorTypes<int, int64_t>>::call(this, Input(1));
        } else {
          return DoRunWithType<int64_t>();
        }
        */
    }
    
    #[inline] pub fn do_run_with_type<SIndex>(&mut self) -> bool {
        todo!();
        /*
            if (InputSize() > 1) {
          ReinitializeAndCopyFrom(
              &starts_host_, at::dtype<SIndex>().device(CPU), Input(1));
          ReinitializeAndCopyFrom(
              &ends_host_, at::dtype<SIndex>().device(CPU), Input(2));
        } else {
          if (!statically_inited_) {
            if (HasArgument("dim") && HasArgument("start_idx") &&
                HasArgument("end_idx")) {
              auto dim = this->template GetSingleArgument<int>("dim", 0);
              auto start =
                  this->template GetSingleArgument<int64_t>("start_idx", 0);
              auto end = this->template GetSingleArgument<int64_t>("end_idx", -1);
              auto& input_tensor = Inputs()[0]->Get<Int8TensorCPU>();
              auto rank = input_tensor.t.sizes().size();
              starts_.resize(rank, 0);
              ends_.resize(rank, -1);
              starts_[dim] = start;
              ends_[dim] = end;
            } else {
              CAFFE_ENFORCE(HasArgument("starts"));
              CAFFE_ENFORCE(HasArgument("ends"));
            }
            CAFFE_ENFORCE_EQ(starts_.size(), ends_.size());

            ReinitializeTensor(
                &starts_host_,
                {static_cast<int64_t>(starts_.size())},
                at::dtype<SIndex>().device(CPU));
            ReinitializeTensor(
                &ends_host_,
                {static_cast<int64_t>(ends_.size())},
                at::dtype<SIndex>().device(CPU));

            memcpy(
                starts_host_.template mutable_data<SIndex>(),
                starts_.data(),
                sizeof(SIndex) * starts_.size());
            memcpy(
                ends_host_.template mutable_data<SIndex>(),
                ends_.data(),
                sizeof(SIndex) * ends_.size());
            statically_inited_ = true;
          }
        }

        auto& X = Inputs()[0]->Get<Int8TensorCPU>();
        auto* Y = Outputs()[0]->GetMutable<Int8TensorCPU>();
        int32_t Y_offset = this->template GetSingleArgument<int>("Y_zero_point", 0);
        auto Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
        CHECK_EQ(Y_offset, X.zero_point);
        CHECK_EQ(Y_scale, X.scale);
        Y->scale = Y_scale;
        Y->zero_point = Y_offset;

        return SliceImpl<SIndex, CPUContext>(
            &Y->t, X.t, starts_host_, ends_host_, &context_);
        */
    }
}
