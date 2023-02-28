crate::ix!();

use crate::{
    OperatorStorage,
    OperatorDef,
    GradientMakerBase,
    CPUContext,
    Tensor
};

/**
  | Constructs a tensor by tiling a given
  | tensor along a specified axis.
  | 
  | This operation creates a new tensor
  | by replicating the input tensor a number
  | of times specified by the `tiles` argument
  | along the `axis` dimension.
  | 
  | The output tensor's `axis` dimension
  | has $(X.dims(axis) * tiles)$ elements.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tile_op.cc
  | 
  | Copy a Blob n times along a specified
  | axis.
  |
  */
pub struct TileOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,

    tiles:   i32,
    axis:    i32,
}

num_inputs!{Tile, (1,3)}

num_outputs!{Tile, 1}

inputs!{Tile, 
    0 => ("X",     "(*Tensor*): input tensor"),
    1 => ("tiles", "(*Tensor`<int>`*): [OPTIONAL] number of replicas (overrides `tiles` argument)"),
    2 => ("axis",  "(*Tensor`<int>`*): [OPTIONAL] axis to replicate along (overrides `axis` argument)")
}

outputs!{Tile, 
    0 => ("Y", "(*Tensor*): output tensor")
}

args!{Tile, 
    0 => ("tiles", "(*int*): number of replicas"),
    1 => ("axis",  "(*int*): axis to replicate along")
}

tensor_inference_function!{
    Tile, 
    /* ([](const OperatorDef& def,
                                const std::vector<TensorShape>& in) {
      std::vector<TensorShape> out(1);
      out[0] = TensorShape(in[0]);
      ArgumentHelper helper(def);
      const std::int32_t tiles =
          helper.GetSingleArgument<std::int32_t>("tiles", 1);
      const std::int32_t axis =
          helper.GetSingleArgument<std::int32_t>("axis", 0);
      if (in.size() > 1) {
        // Tile or axis is specified as input; we can't determine
        // the size
        out[0].set_unknown_shape(true);
      } else {
        const auto canonical_axis =
            canonical_axis_index_(axis, out[0].dims().size());
        out[0].set_dims(
            canonical_axis, out[0].dims().Get(canonical_axis) * tiles);
      }
      return out;
    }) */
}

inherit_onnx_schema!{Tile}

register_cpu_operator!{Tile, TileOp<CPUContext>}

#[test] fn tile_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Tile",
        ["X", "tiles", "axis"],
        ["Y"]
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(5,5)))
    workspace.FeedBlob("tiles", np.array([5]).astype(np.int32))
    workspace.FeedBlob("axis", np.array([1]).astype(np.int32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))


    X:
    [[9 1 7 1 3]
     [2 3 6 2 5]
     [0 9 2 6 4]
     [5 8 1 5 9]
     [2 0 1 3 7]]
    Y:
    [[9 1 7 1 3 9 1 7 1 3 9 1 7 1 3 9 1 7 1 3 9 1 7 1 3]
     [2 3 6 2 5 2 3 6 2 5 2 3 6 2 5 2 3 6 2 5 2 3 6 2 5]
     [0 9 2 6 4 0 9 2 6 4 0 9 2 6 4 0 9 2 6 4 0 9 2 6 4]
     [5 8 1 5 9 5 8 1 5 9 5 8 1 5 9 5 8 1 5 9 5 8 1 5 9]
     [2 0 1 3 7 2 0 1 3 7 2 0 1 3 7 2 0 1 3 7 2 0 1 3 7]]

    */
}

impl<Context> TileOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(std::int32_t, "tiles", tiles_, 1),
            OP_SINGLE_ARG(std::int32_t, "axis", axis_, 0)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<
            TensorTypes<std::int32_t, std::int64_t, float, double>>::
            call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            if (InputSize() > 1) {
          // We potentially have tiles and/or axis specified as inputs
          // as well. We will check for them in that order. In other words:
          // InputSize() == 2: tiles is specified
          // InputSize() == 3: tiles is specified and axis.
          // Anything specified as input will override the arguments
          CAFFE_ENFORCE(
              Input(1).dim() == 1 && Input(1).numel() == 1,
              "Input `tiles` should be a vector of size 1.");
          tiles_ = GetArgFromTensor(Input(1));

          // Because of a bug in original code, temporarily adds this part to keep
          // backward compatibility.
          // TODO(yangxm): Remove this part when prod runtime upgraded with fixed
          // model config.
          if (Input(1).template IsType<std::int64_t>()) {
            axis_ = 0;
          }

          if (InputSize() > 2) {
            CAFFE_ENFORCE(
                Input(2).dim() == 1 && Input(2).numel() == 1,
                "Input `axis` should be a vector of size 1.");
            axis_ = GetArgFromTensor(Input(2));
          } else {
            CAFFE_ENFORCE(
                OperatorStorage::HasArgument("axis"),
                "Argument `axis` is missing and was not specified as input.");
          }
        } else {
          CAFFE_ENFORCE(
              OperatorStorage::HasArgument("tiles"),
              "Argument `tiles` is missing and was not specified as input.");
          CAFFE_ENFORCE(
              OperatorStorage::HasArgument("axis"),
              "Argument `axis` is missing and was not specified as input.");
        }

        const auto& X = Input(0);
        auto* Y = Output(0);
        const int axis = X.canonical_axis_index(axis_);

        // reshape output to be input tiled along the axis
        std::vector<std::int64_t> Y_dims = X.sizes().vec();
        Y_dims[axis] *= tiles_;
        Y->Resize(Y_dims);

        // size up to (and not including) axis
        const int outer_size = X.size_to_dim(axis);
        // size from axis up
        const int inner_size = X.size_from_dim(axis);

        const T* X_data = X.template data<T>();
        T* Y_data = Y->template mutable_data<T>();
        return DoTile<T>(outer_size, inner_size, X_data, Y_data);
        */
    }
    
    #[inline] pub fn get_arg_from_tensor(&mut self, tensor: &Tensor) -> i32 {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            tensor.IsType<std::int32_t>() || tensor.IsType<std::int64_t>());
        std::int32_t val = -1;
        if (tensor.IsType<std::int32_t>()) {
          context_.template CopyToCPU<std::int32_t>(
              1, tensor.data<std::int32_t>(), &val);
        } else if (tensor.IsType<std::int64_t>()) {
          std::int64_t val_int64;
          context_.template CopyToCPU<std::int64_t>(
              1, tensor.data<std::int64_t>(), &val_int64);
          val = static_cast<std::int32_t>(val_int64);
        }
        return val;
        */
    }
    
    #[inline] pub fn do_tile<T>(&mut self, 
        outer_size: i32,
        inner_size: i32,
        x:          *const T,
        y:          *mut T) -> bool {

        todo!();
        /*
            if (inner_size == 1) {
          EigenArrayMap<T> Y_arr(Y, tiles_, outer_size);
          for (int i = 0; i < outer_size; ++i) {
            Y_arr.col(i) = X[i];
          }
        } else {
          ConstEigenArrayMap<T> X_arr(X, inner_size, outer_size);
          for (int i = 0; i < outer_size; ++i) {
            EigenArrayMap<T>(Y + i * tiles_ * inner_size, inner_size, tiles_)
                .colwise() = X_arr.col(i);
          }
        }
        return true;
        */
    }
}

impl TileOp<CPUContext> {

    #[inline] pub fn run_on_cpu_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<
          at::Half,
          std::uint8_t,
          std::int32_t,
          std::int64_t,
          float,
          double,
          std::string>>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type_string(&mut self) -> bool {
        
        todo!();
        /*
            if (InputSize() > 1) {
        // We potentially have tiles and/or axis specified as inputs
        // as well. We will check for them in that order. In other words:
        // InputSize() == 2: tiles is specified
        // InputSize() == 3: tiles is specified and axis.
        // Anything specified as input will override the arguments
        CAFFE_ENFORCE(
            Input(1).dim() == 1 && Input(1).numel() == 1,
            "Input `tiles` should be a vector of size 1.");
        tiles_ = GetArgFromTensor(Input(1));

        // Because of a bug in original code, temporarily adds this part to keep
        // backward compatibility.
        // TODO(yangxm): Remove this part when prod runtime upgraded with fixed
        // model config.
        if (Input(1).IsType<std::int64_t>()) {
          axis_ = 0;
        }

        if (InputSize() > 2) {
          CAFFE_ENFORCE(
              Input(2).dim() == 1 && Input(2).numel() == 1,
              "Input `axis` should be a vector of size 1.");
          axis_ = GetArgFromTensor(Input(2));
        } else {
          CAFFE_ENFORCE(
              OperatorStorage::HasArgument("axis"),
              "Argument `axis` is missing and was not specified as input.");
        }
      } else {
        CAFFE_ENFORCE(
            OperatorStorage::HasArgument("tiles"),
            "Argument `tiles` is missing and was not specified as input.");
        CAFFE_ENFORCE(
            OperatorStorage::HasArgument("axis"),
            "Argument `axis` is missing and was not specified as input.");
      }

      const auto& X = Input(0);
      auto* Y = Output(0);
      const int axis = X.canonical_axis_index(axis_);

      // reshape output to be input tiled along the axis
      std::vector<std::int64_t> Y_dims = X.sizes().vec();
      Y_dims[axis] *= tiles_;
      Y->Resize(Y_dims);

      // size up to (and not including) axis
      const int outer_size = X.size_to_dim(axis);
      // size from axis up
      const int inner_size = X.size_from_dim(axis);

      const TypeMeta meta = X.dtype();
      const int item_size = X.itemsize();
      const char* X_ptr = reinterpret_cast<const char*>(X.raw_data());
      char* Y_ptr = reinterpret_cast<char*>(Y->raw_mutable_data(meta));
      for (int i = 0; i < outer_size; ++i) {
        for (int t = 0; t < tiles_; ++t) {
          context_.CopyItemsSameDevice(meta, inner_size, X_ptr, Y_ptr);
          Y_ptr += inner_size * item_size;
        }
        X_ptr += inner_size * item_size;
      }
      return true;
        */
    }
}

///-----------------------------------
pub struct TileGradientOp<Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS
    storage: OperatorStorage,
    context: Context,
    tiles:   i32,
    axis:    i32,
    ones:    Tensor,
}

num_inputs!{TileGradient, (1,3)}

num_outputs!{TileGradient, 1}

register_cpu_operator!{TileGradient, TileGradientOp<CPUContext>}

impl<Context> TileGradientOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(std::int32_t, "tiles", tiles_, 1),
            OP_SINGLE_ARG(std::int32_t, "axis", axis_, 0)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<
            TensorTypes<std::int32_t, std::int64_t, float, double>>::
            call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            if (InputSize() > 1) {
          // We potentially have tiles and/or axis specified as inputs
          // as well. We will check for them in that order. In other words:
          // InputSize() == 2: tiles is specified
          // InputSize() == 3: tiles is specified and axis.
          // Anything specified as input will override the arguments
          CAFFE_ENFORCE(
              Input(1).dim() == 1 && Input(1).numel() == 1,
              "Input `tiles` should be a vector of size 1.");
          tiles_ = GetArgFromTensor(Input(1));
          if (InputSize() > 2) {
            CAFFE_ENFORCE(
                Input(2).dim() == 1 && Input(2).numel() == 1,
                "Input `axis` should be a vector of size 1.");
            axis_ = GetArgFromTensor(Input(2));
          } else {
            CAFFE_ENFORCE(
                OperatorStorage::HasArgument("axis"),
                "Argument `axis` is missing and was not specified as input.");
          }
        } else {
          CAFFE_ENFORCE(
              OperatorStorage::HasArgument("tiles"),
              "Argument `tiles` is missing and was not specified as input.");
          CAFFE_ENFORCE(
              OperatorStorage::HasArgument("axis"),
              "Argument `axis` is missing and was not specified as input.");
        }

        const auto& dY = Input(0);
        auto* dX = Output(0);
        const int axis = dY.canonical_axis_index(axis_);

        // reshape output to be input "untiled" along the axis
        std::vector<std::int64_t> X_dims = dY.sizes().vec();
        CAFFE_ENFORCE_EQ(X_dims[axis] % tiles_, 0);
        X_dims[axis] /= tiles_;
        dX->Resize(X_dims);

        // size up to (and not including) axis
        const int outer_size = dX->size_to_dim(axis);
        // size from axis up
        const int inner_size = dX->size_from_dim(axis);

        /**
         * How this works:
         * Imagine a 2D tensor (matrix) of size 3x10, tiled 2 times along axis 1
         * (column).
         * This is equivalent to multiplying by a vector of 1s transposed.
         * The gradient of this is all 1s in the shape of the input matrix
         * (call it X).
         * So the output gradient should be the matrix multiplication result
         * of input gradient (gradient of tiled tensor output) and X.
         */
        const T* dY_data = dY.template data<T>();
        T* dX_data = dX->template mutable_data<T>();
        return DoTileGradient<T>(outer_size, inner_size, dY_data, dX_data);
        */
    }
    
    #[inline] pub fn get_arg_from_tensor(&mut self, tensor: &Tensor) -> i32 {
        
        todo!();
        /*
            CAFFE_ENFORCE(
            tensor.IsType<std::int32_t>() || tensor.IsType<std::int64_t>());
        std::int32_t val = -1;
        if (tensor.IsType<std::int32_t>()) {
          context_.template CopyToCPU<std::int32_t>(
              1, tensor.data<std::int32_t>(), &val);
        } else if (tensor.IsType<std::int64_t>()) {
          std::int64_t val_int64;
          context_.template CopyToCPU<std::int64_t>(
              1, tensor.data<std::int64_t>(), &val_int64);
          val = static_cast<std::int32_t>(val_int64);
        }
        return val;
        */
    }
    
    #[inline] pub fn do_tile_gradient<T>(&mut self, 
        outer_size: i32,
        inner_size: i32,
        dy:         *const T,
        dx:         *mut T) -> bool {

        todo!();
        /*
            if (inner_size == 1) {
          const std::array<int, 2> dY_dims = {outer_size, tiles_};
          const std::array<int, 2> dX_dims = {outer_size, 1};
          math::ReduceSum<T, Context>(
              2, dY_dims.data(), dX_dims.data(), T(1), dY, dX, &context_);
        } else {
          math::CopyMatrix<T, Context>(
              outer_size,
              inner_size,
              dY,
              inner_size * tiles_,
              dX,
              inner_size,
              &context_);
          for (int i = 0; i < outer_size; ++i) {
            const T* dY_ptr = dY + i * tiles_ * inner_size;
            T* dX_ptr = dX + i * inner_size;
            for (int j = 1; j < tiles_; ++j) {
              math::Add<T, Context>(
                  inner_size, dX_ptr, dY_ptr + j * inner_size, dX_ptr, &context_);
            }
          }
        }
        return true;
        */
    }
}

///---------------------
pub struct GetTileGradient<'a> {
    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetTileGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            // Check whether the tiles/axis information was
        // passed through input arguments
        std::vector<std::string> g_inputs({GO(0)});
        if (Def().input_size() > 1) {
          g_inputs.push_back(I(1));
        }
        if (Def().input_size() > 2) {
          g_inputs.push_back(I(2));
        }
        return SingleGradientDef(
            "TileGradient", "", g_inputs, std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{Tile, GetTileGradient}
