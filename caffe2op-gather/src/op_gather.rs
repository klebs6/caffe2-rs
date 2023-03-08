crate::ix!();

#[test] fn gather_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Gather",
        ["DATA", "INDICES"],
        ["OUTPUT"]
    )
    data = np.array([[1., 1.2],[2.3, 3.4],[4.5, 5.7]])
    print("DATA:\n",data)

    inds = np.array([[0, 1],[1, 2]])
    print("INDICES:\n",inds)

    // Feed X into workspace
    workspace.FeedBlob("DATA", data.astype(np.float32))
    workspace.FeedBlob("INDICES", inds.astype(np.int32))

    workspace.RunOperatorOnce(op)
    print("OUTPUT:\n", workspace.FetchBlob("OUTPUT"))

    DATA:
     [[1.  1.2]
     [2.3 3.4]
     [4.5 5.7]]
    INDICES:
     [[0 1]
     [1 2]]
    OUTPUT:
     [[[1.  1.2]
      [2.3 3.4]]

     [[2.3 3.4]
      [4.5 5.7]]]

    */
}

/**
  | The *Gather* op accepts a *DATA* tensor
  | of rank $r >= 1$ and *INDICES* tensor
  | of rank $q$ as inputs.
  | 
  | It then gathers entries of the outer-most
  | dimension of *DATA*, indexed by *INDICES*,
  | and concatenate them in an output tensor
  | of rank $q + (r - 1)$.
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/gather_op.cc
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/gather_op.h
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GatherOp<Context> {

    storage:      OperatorStorage,
    context:      Context,

    axis:         i32,
    wrap_indices: bool,
    match_outer:  bool,
}

register_cpu_operator!{Gather, GatherOp<CPUContext>}

num_inputs!{Gather, 2}

num_outputs!{Gather, 1}

inputs!{Gather, 
    0 => ("DATA", "Input data tensor of rank $r>=1$"),
    1 => ("INDICES", "Input indices tensor of rank $q$. This tensor must contain integers.")
}

outputs!{Gather, 
    0 => ("OUTPUT", "Output tensor of rank $q+(r-1)$")
}

tensor_inference_function!{Gather, 
    |def: &OperatorDef, input: &Vec<TensorShape>| {
        todo!();
        /*
          ArgumentHelper helper(def);
          const int axis = helper.GetSingleArgument<int>("axis", 0);
          const bool match_outer =
              helper.GetSingleArgument<bool>("match_outer", false);
          const auto& data_dims = GetDimsVector(in[0]);
          const auto& indices_dims = GetDimsVector(in[1]);

          vector<int> output_dims =
              caffe2::gather_helper::calc_output_shape_vector<int>(
                  data_dims, indices_dims, axis, match_outer);
          vector<TensorShape> out(1);
          out[0] = CreateTensorShape(output_dims, in[0].data_type());
          return out;
        */
    }
}

inherit_onnx_schema!{Gather}

input_tags!{
    GatherOp {
        Data,
        Indices
    }
}

impl<Context> GatherOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "axis", axis_, 0),
            OP_SINGLE_ARG(bool, "match_outer", match_outer_, false) 

        // TBD: We may want to fix the old index wrap behaviour once we have
        // operator versioning, to only apply it when needed as otherwise its likely
        // an error.
        // Right now, we apply index wrapping by default only to axis == 0,
        // since we have ONNX conversion code that uses it. For other ops it
        // needs to be specified explicitly with argument or you don't get it.
        if (OperatorStorage::HasArgument("wrap_indices")) {
          wrap_indices_ = Operator<Context>::template GetSingleArgument<bool>(
              "wrap_indices", (false));
        } else {
          wrap_indices_ = (axis_ == 0) ? true : false;
        }
        */
    }
    
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
            return gather_helper::gather_impl<Index, Context>(
                this, DATA, INDICES, 0, axis_, wrap_indices_, match_outer_);
        */
    }
}

pub struct GetGatherGradient;

impl GetGradientDefs for GetGatherGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            ArgumentHelper argsHelper(def_);
        const bool dense_gradient =
            argsHelper.GetSingleArgument<bool>("dense_gradient", false);
        const int axis = argsHelper.GetSingleArgument<int>("axis", 0);

        // TBD: While it hasn't been used yet, we need to add wrap_indices support
        // to gradients next.
        // if (argsHelper.HasArgument("wrap_indices_")) {
        // }

        using Op = GatherOp<CPUContext>;

        if (axis == 0) {
          if (dense_gradient) {
            return vector<OperatorDef>{CreateOperatorDef(
                "SparseToDense",
                "",
                vector<string>{I(Op::INDICES), GO(0), I(Op::DATA)},
                vector<string>{GI(Op::DATA)})};
          } else {
            // For now we don't do any reshaping as the consumer of this op would
            // probably be ScatterUpdate which is intenionally ignores shapes. We
            // might need to revisit it in the future for correctness purposes. The
            // right shape for the output woild be to flatten INDICES and collapse
            // first X dims of GRAD
            SetSparse(Op::DATA, I(Op::INDICES), GO(0));
            return vector<OperatorDef>();
          }
        }

        // TBD: This is misleading to use dense_gradient by default for axis 0
        // and not othewise....
        if (argsHelper.HasArgument("dense_gradient")) {
          CAFFE_ENFORCE(
              dense_gradient == true,
              "Gather with axis > 0 must use dense_gradient");
        }

        Argument axisArg = MakeArgument<int>("axis", axis);
        return SingleGradientDef(
            "BatchGatherGradient",
            "",
            // This is the order as expected by BatchGatherGradient indices,
            // different from SpartseToDense above.
            vector<string>{I(Op::DATA), I(Op::INDICES), GO(0)},
            vector<string>{GI(0)},
            std::vector<Argument>{axisArg});
        */
    }
}

register_gradient!{Gather, GetGatherGradient}

/**
  | New shape is concatenation:
  |
  |  [data dims before axis] + [indices dims]
  |  + [data dims after axis]
  */
#[inline] pub fn calc_output_shape_vector<IndexType, DataDimsVec, IndexDimsVec>(
    data_dims:    &DataDimsVec,
    indices_dims: &IndexDimsVec,
    axis:         i32,
    match_outer:  bool) -> Vec<IndexType> 
{
    todo!();
    /*
        vector<IndexType> shape;
      // If the dimension we are indexing is empty, just use data_dims as shape.
      // This replicates behavior in (https://github.com/pytorch/pytorch/pull/13781)
      // needed to allow workflows with empty batch to succeed.
      if (data_dims[axis] == 0) {
        shape.insert(shape.end(), data_dims.begin(), data_dims.end());
      } else {
        shape.insert(shape.end(), data_dims.begin(), data_dims.begin() + axis);
        if (match_outer) {
          shape.insert(
              shape.end(), indices_dims.begin() + axis, indices_dims.end());
        } else {
          shape.insert(shape.end(), indices_dims.begin(), indices_dims.end());
        }
        shape.insert(shape.end(), data_dims.begin() + axis + 1, data_dims.end());
      }
      return shape;
    */
}

/**
  | Check that indices fall within dimension
  | array size with CAFFE_ENFORCE.
  |
  */
#[inline] pub fn check_indexarray_range<IndexType>(
    indices:           *const IndexType,
    n:                 i64,
    indexing_axis_dim: IndexType,
    wrap_indices:      bool) 
{
    todo!();
    /*
        //
      for (auto i = 0; i < n; ++i) {
        auto idx = indices[i];
        if (wrap_indices && idx < 0) {
          idx = idx + indexing_axis_dim;
        }
        CAFFE_ENFORCE(
            0 <= idx && idx < indexing_axis_dim,
            "INDICES element is out of DATA bounds, id=",
            idx,
            " axis_dim=",
            indexing_axis_dim);
      }
    */
}

/**
  | Actual gather implementation - resizes
  | output and copies indexed data.
  |
  */
#[inline] pub fn gather_impl<Index, Context>(
    op:            *mut dyn Operator,
    data_idx:      i32,
    indices_idx:   i32,
    output_idx:    i32,
    axis:          i32,
    wrap_indices:  bool,
    match_outer:   bool) -> bool 
{
    todo!();
    /*
        // If we endup using it on GPU doing O(N) memcpy is probably not best :)
      // TODO: implement prefetching if it starts mattering (TF does it)

      const Tensor& data = op->Input(dataIdx);
      const Tensor& indices = op->Input(indicesIdx);
      const TypeMeta dataType = data.dtype();
      size_t item_bytesize = dataType.itemsize();

      // ONNX allows negative axis to index from the back, valid range: [-r, r].
      if (axis < 0) {
        axis = data.dim() + axis;
      }
      CAFFE_ENFORCE_GE(data.dim(), axis + 1, "DATA should be at least [axis+1]-D");
      CAFFE_ENFORCE_GE(axis, 0, "Axis should be non-negative");
      CAFFE_ENFORCE_LT(axis, data.dim(), "Axis out of range");

      // New shape:
      //  [data dims before axis] + [indices dims] + [data dims after axis]
      vector<int64_t> shape = calc_output_shape_vector<int64_t>(
          data.sizes(), indices.sizes(), axis, match_outer);
      Tensor* output = op->Output(outputIdx, shape, at::dtype(dataType));
      auto out = static_cast<char*>(output->raw_mutable_data(dataType));

      // Succeed if size of output is zero, which can happen for empty batch which
      // would have data dimension size of 0.
      // This *must* be done AFTER output->raw_mutable_data() above as that has
      // important allocation side effect that we must see.
      if (output->numel() == 0) {
        return true;
      }

      const Index* idxs = indices.template data<Index>();
      auto src_base = static_cast<const char*>(data.raw_data());

      auto outer_dims_product = data.size_to_dim(axis);
      auto block_size = data.size_from_dim(axis + 1);
      auto block_bytesize = block_size * item_bytesize;

      auto src_indexing_axis_dim = data.size(axis);
      auto src_batch_bytesize = data.size_from_dim(axis) * item_bytesize;
      // Treat indices as a single block even if they have multiple dimensions.
      // The "gathered batch" is a cumulative result combining indexed blocks.
      auto idx_inner_dims_product = indices.size_from_dim(axis);
      auto N = indices.numel();
      if (match_outer) {
        CAFFE_ENFORCE_GE(axis, 1, "Axis should be at least 1");
        for (auto i = 0; i < axis; i++) {
          CAFFE_ENFORCE_EQ(
              data.size(i),
              indices.size(i),
              "INDICES must have the same outer dims as DATA (before dim AXIS)");
        }
        N = idx_inner_dims_product;
      }

      auto gathered_batch_bytesize = N * block_size * item_bytesize;

      check_indexarray_range<Index>(idxs, N, src_indexing_axis_dim, wrap_indices);

      // Special-case single-float copy for efficiency
      if (data.template IsType<float>() && block_size == 1) {
        for (auto batch = 0; batch < outer_dims_product; ++batch) {
          const float* src_floats =
              (const float*)(src_base + batch * src_batch_bytesize);
          float* dst_floats = (float*)(out + batch * gathered_batch_bytesize);

          for (auto i = 0; i < N; ++i) {
            auto idx = idxs[i];
            if (match_outer) {
              idx = idxs[batch * idx_inner_dims_product + i];
            }
            if (wrap_indices && idx < 0) {
              idx = idx + src_indexing_axis_dim;
            }
            dst_floats[i] = src_floats[idx];
          }
        }
      } else {
        // outer_dims_product specifies how many times we repeat inner dimensions,
        // so we just iterate over it to cover all outer dimensions.
        for (auto batch = 0; batch < outer_dims_product; ++batch) {
          for (auto i = 0; i < N; ++i) {
            auto idx = idxs[i];
            if (match_outer) {
              idx = idxs[batch * idx_inner_dims_product + i];
            }
            if (wrap_indices && idx < 0) {
              idx = idx + src_indexing_axis_dim;
            }

            auto src = src_base + batch * src_batch_bytesize + idx * block_bytesize;
            auto dst = out + batch * gathered_batch_bytesize + i * block_bytesize;
            op->getContext()->CopyItemsSameDevice(dataType, block_size, src, dst);
          }
        }
      }
      return true;
    */
}
