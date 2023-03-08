crate::ix!();

use crate::{
    OperatorStorage,
    GradientMakerBase,
    Tensor,
    OperatorDef
};

#[inline] pub fn slice_impl<SIndex, Context>(
    output:  *mut Tensor,
    data:    &Tensor,
    starts:  &Tensor,
    ends:    &Tensor,
    context: *mut Context,
    gdata:   Option<&mut Tensor>,
    go:      Option<&Tensor>) -> bool {

    todo!();
    /*
        bool backward = output == nullptr;

      auto* starts_data = starts.template data<SIndex>();
      auto* ends_data = ends.template data<SIndex>();

      CAFFE_ENFORCE_EQ(starts.dim(), 1);
      CAFFE_ENFORCE_EQ(ends.dim(), 1);
      CAFFE_ENFORCE_GE(data.dim(), starts.numel());
      CAFFE_ENFORCE_EQ(starts.numel(), ends.numel());

      std::vector<SIndex> starts_idx(data.dim());
      std::vector<SIndex> ends_idx(data.dim());
      std::vector<SIndex> dst_sizes(data.dim());

      for (int i = 0; i < data.dim(); ++i) {
        if (i >= starts.numel()) {
          starts_idx[i] = 0;
          ends_idx[i] = data.size(i);
          dst_sizes[i] = data.size(i);
          continue;
        }
        if (data.size(i) > 0) {
          auto start = starts_data[i];
          auto end = ends_data[i];
          if (start < 0) {
            start = data.size(i) + 1 + start;
          }
          if (end < 0) {
            end = data.size(i) + 1 + end;
          }
          if (start > data.size(i)) {
            start = data.size(i);
          }
          if (end > data.size(i)) {
            end = data.size(i);
          }
          CAFFE_ENFORCE_GE(start, 0);
          CAFFE_ENFORCE_GE(end, 0);
          CAFFE_ENFORCE_GE(end, start);
          starts_idx[i] = start;
          ends_idx[i] = end;
          dst_sizes[i] = end - start;
        } else {
          starts_idx[i] = 0;
          ends_idx[i] = 0;
          dst_sizes[i] = 0;
        }
      }

      if (data.numel() <= 0) {
        // When the input is empty, we do not need to do copy.
        if (!backward) {
          output->Resize(dst_sizes);
          output->raw_mutable_data(data.dtype());
        } else {
          gdata->ResizeLike(data);
          gdata->raw_mutable_data(go->dtype());
        }
        return true;
      }
      // for now only supports slicing in 1 dimension
      int dim = -1;
      for (int i = 0; i < data.dim(); ++i) {
        if (starts_idx[i] > 0 || ends_idx[i] < data.size(i)) {
          CAFFE_ENFORCE_EQ(
              dim, -1, "Currently only possible to slice in 1 dimension.");
          dim = i;
        }
      }
      if (dim == -1) {
        if (!backward) {
          output->CopyFrom(data, true /*async*/);
        } else {
          gdata->CopyFrom(*go, true /*async*/);
        }
        return true;
      }
      size_t unit = std::accumulate(
          data.sizes().begin() + dim + 1,
          data.sizes().end(),
          1,
          std::multiplies<SIndex>());
      size_t num_blocks = std::accumulate(
          data.sizes().begin(),
          data.sizes().begin() + dim,
          1,
          std::multiplies<SIndex>());
      if (!backward) {
        output->Resize(dst_sizes);
      } else {
        gdata->ResizeLike(data);
      }

      size_t itemsize = data.dtype().itemsize();

      if (!backward) {
        char* src_bytes = (char*)data.raw_data();
        char* dst_bytes = (char*)output->raw_mutable_data(data.dtype());

        size_t src_nbytes = data.nbytes();
        size_t dst_nbytes = output->nbytes();

        size_t src_block_size = unit * data.size(dim);
        size_t dst_block_size = unit * (ends_idx[dim] - starts_idx[dim]);
        size_t src_offset = unit * starts_idx[dim];

        if (num_blocks == 0 || dst_block_size == 0) {
          return true;
        }

        size_t src_block_size_bytes = itemsize * src_block_size;
        size_t dst_block_size_bytes = itemsize * dst_block_size;

        char* src_offset_bytes = src_bytes + itemsize * src_offset;
        char* dst_offset_bytes = dst_bytes;
        for (size_t i = 0; i < num_blocks; ++i) {
          char* local_src_offset_bytes =
              src_offset_bytes + i * src_block_size_bytes;
          char* local_dst_offset_bytes =
              dst_offset_bytes + i * dst_block_size_bytes;
          DCHECK_LE(
              static_cast<void*>(local_src_offset_bytes + dst_block_size_bytes),
              static_cast<void*>(src_bytes + src_nbytes));
          DCHECK_LE(
              static_cast<void*>(local_dst_offset_bytes + dst_block_size_bytes),
              static_cast<void*>(dst_bytes + dst_nbytes));
          context->CopyItemsSameDevice(
              data.dtype(),
              dst_block_size,
              (void*)local_src_offset_bytes,
              (void*)local_dst_offset_bytes);
        }
      } else {
        char* src_bytes = (char*)go->raw_data();
        char* dst_bytes = (char*)gdata->raw_mutable_data(go->dtype());

        size_t src_nbytes = go->nbytes();
        size_t dst_nbytes = gdata->nbytes();

        size_t src_block_size = unit * (ends_idx[dim] - starts_idx[dim]);
        size_t dst_block_size = unit * data.size(dim);
        size_t dst_offset = unit * starts_idx[dim];

        if (num_blocks == 0 || dst_block_size == 0) {
          return true;
        }

        size_t src_block_size_bytes = itemsize * src_block_size;
        size_t dst_block_size_bytes = itemsize * dst_block_size;

        char* src_offset_bytes = src_bytes;
        char* dst_offset_bytes = dst_bytes + itemsize * dst_offset;
        // Zero out gradient blob before copy since we copy in fewer items than
        // there is space for
        math::Set<char, Context>(dst_nbytes, 0, dst_bytes, context);

        // If output tensor is empty, just return zeroed gradient tensor
        if (!src_bytes) {
          return true;
        }

        for (size_t i = 0; i < num_blocks; ++i) {
          char* local_src_offset_bytes =
              src_offset_bytes + i * src_block_size_bytes;
          char* local_dst_offset_bytes =
              dst_offset_bytes + i * dst_block_size_bytes;
          DCHECK_LE(
              local_src_offset_bytes + src_block_size_bytes,
              src_bytes + src_nbytes);
          DCHECK_LE(
              local_dst_offset_bytes + src_block_size_bytes,
              dst_bytes + dst_nbytes);
          context->CopyItemsSameDevice(
              go->dtype(),
              src_block_size,
              (void*)local_src_offset_bytes,
              (void*)local_dst_offset_bytes);
        }
      }
      return true;
    */
}

///----------------------
#[test] fn slice_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Slice",
        ["X"],
        ["Y"],
        starts=(0,1),
        ends=(-1,3)
    )

    workspace.FeedBlob("X", np.array([[1,2,3,4],[5,6,7,8]]))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))


    X:
    [[1 2 3 4]
     [5 6 7 8]]
    Y:
    [[2 3]
     [6 7]]
    */
}

/**
  | Produces a slice of the input tensor.
  | 
  | - Currently, only slicing in a single
  | dimension is supported.
  | 
  | - Start and end indices are either passed
  | as two 1D input tensors or using the `starts`
  | and `ends` arguments.
  | 
  | - If a negative value is passed for any
  | of the start or end indices, it represents
  | |value| - 1 elements before the end of
  | that dimension. End indices are non-inclusive
  | unless negative (end index
  | 
  | -1 means up to and including the last
  | element).
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/slice_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SliceOp<Context> {

    storage:            OperatorStorage,
    context:            Context,

    starts:             Vec<i64>,
    ends:               Vec<i64>,
    statically_inited:  bool,
    starts_host:        Tensor,
    ends_host:          Tensor,
}

register_cpu_operator!{Slice, SliceOp<CPUContext>}

num_inputs!{Slice, (1,3)}

num_outputs!{Slice, 1}

inputs!{Slice, 
    0 => ("X", "(*Tensor*): tensor to extract slices from"),
    1 => ("starts", "(*Tensor`<int>`*): 1D tensor of start-indices for each dimension of data (dimensions following the sliced one might be omitted)"),
    2 => ("ends", "(*Tensor`<int>`*): 1D tensor of end-indices for each dimension of data (dimensions following the sliced one might be omitted)")
}

outputs!{Slice, 
    0 => ("Y", "(*Tensor*): sliced output tensor")
}

args!{Slice, 
    0 => ("starts", "(*Tuple(int)*): list of starting indices"),
    1 => ("ends", "(*Tuple(int)*): list of ending indices")
}

/// the filler cannot be enabled without output dims
disallow_input_fillers!{Slice}

tensor_inference_function!{Slice, /* ([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      if (in.size() > 1) {
        // Cannot compute shape inference when the splits are defined
        // in data.
        return vector<TensorShape>();
      }
      auto const& data = in[0];

      ArgumentHelper helper(def);
      auto starts = helper.GetRepeatedArgument<int>("starts", vector<int>());
      auto ends = helper.GetRepeatedArgument<int>("ends", vector<int>());
      vector<int> dst_sizes(data.dims_size());

      for (int i = 0; i < data.dims_size(); ++i) {
        if (i >= starts.size()) {
          dst_sizes[i] = data.dims(i);
          continue;
        }
        if (data.dims(i) > 0) {
          auto start = starts[i];
          auto end = ends[i];
          if (start < 0) {
            start = data.dims(i) + 1 + start;
          }
          if (end < 0) {
            end = data.dims(i) + 1 + end;
          }
          dst_sizes[i] = end - start;
        } else {
          dst_sizes[i] = 0;
        }
      }
      return vector<TensorShape>{
          CreateTensorShape(dst_sizes, data.data_type())};
    }) */}

inherit_onnx_schema!{Slice}

impl<Context> SliceOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            starts_(this->template GetRepeatedArgument<int64_t>("starts")),
            ends_(this->template GetRepeatedArgument<int64_t>("ends")),
            statically_inited_(false)
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
          ReinitializeAndCopyFrom(&starts_host_, at::dtype<SIndex>().device(CPU), Input(1));
          ReinitializeAndCopyFrom(&ends_host_, at::dtype<SIndex>().device(CPU), Input(2));
        } else {
          if (!statically_inited_) {
            CAFFE_ENFORCE(HasArgument("starts"));
            CAFFE_ENFORCE(HasArgument("ends"));
            CAFFE_ENFORCE_EQ(starts_.size(), ends_.size());

            ReinitializeTensor(&starts_host_, {static_cast<int64_t>(starts_.size())}, at::dtype<SIndex>().device(CPU));
            ReinitializeTensor(&ends_host_, {static_cast<int64_t>(ends_.size())}, at::dtype<SIndex>().device(CPU));

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

        const auto& data = Input(0);
        auto output = Output(0);

        return SliceImpl<SIndex, Context>(
            output, data, starts_host_, ends_host_, &context_);
        */
    }
}

///------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SliceGradientOp<Context> {

    storage:            OperatorStorage,
    context:            Context,

    starts:             Vec<i64>,
    ends:               Vec<i64>,
    statically_inited:  bool,
    starts_host:        Tensor,
    ends_host:          Tensor,
}

register_cpu_gradient_operator!{SliceGradient, SliceGradientOp<CPUContext>}

tensor_inference_function!{SliceGradient, 
    /* ([](const OperatorDef& /*def*/,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out.at(0) = in.at(0);
      return out;
    }) */
}

impl<Context> SliceGradientOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            starts_(this->template GetRepeatedArgument<int64_t>("starts")),
            ends_(this->template GetRepeatedArgument<int64_t>("ends")),
            statically_inited_(false)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (InputSize() == 4) {
          return DispatchHelper<TensorTypes<int, int64_t>>::call(this, Input(1));
        } else {
          return DoRunWithType<int64_t>();
        }
        */
    }
    
    #[inline] pub fn do_run_with_type<SIndex>(&mut self) -> bool {
    
        todo!();
        /*
            auto* gdata = Output(0);
        auto& data = Input(0);

        if (InputSize() == 4) {
          ReinitializeAndCopyFrom(&starts_host_, at::dtype<SIndex>().device(CPU), Input(1));
          ReinitializeAndCopyFrom(&ends_host_, at::dtype<SIndex>().device(CPU), Input(2));

          auto& go = Input(3);

          return SliceImpl<SIndex, Context>(
              nullptr, data, starts_host_, ends_host_, &context_, gdata, &go);
        } else {
          if (!statically_inited_) {
            CAFFE_ENFORCE(HasArgument("starts"));
            CAFFE_ENFORCE(HasArgument("ends"));
            CAFFE_ENFORCE_EQ(starts_.size(), ends_.size());

            ReinitializeTensor(
                &starts_host_, {static_cast<int64_t>(starts_.size())}, at::dtype<SIndex>().device(CPU));
            ReinitializeTensor(
                &ends_host_, {static_cast<int64_t>(ends_.size())}, at::dtype<SIndex>().device(CPU));

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
          auto& go = Input(1);

          return SliceImpl<SIndex, Context>(
              nullptr, data, starts_host_, ends_host_, &context_, gdata, &go);
        }
        */
    }
}

pub struct GetSliceGradient {

}

impl GetGradientDefs for GetSliceGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            if (def_.input_size() > 1) {
          return vector<OperatorDef>{CreateOperatorDef(
              "SliceGradient",
              "",
              std::vector<string>{I(0), I(1), I(2), GO(0)},
              std::vector<string>{GI(0)})};
        } else {
          return vector<OperatorDef>{CreateOperatorDef(
              "SliceGradient",
              "",
              std::vector<string>{I(0), GO(0)},
              std::vector<string>{GI(0)})};
        }
        */
    }
}

register_gradient!{Slice, GetSliceGradient}
