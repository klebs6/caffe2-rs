crate::ix!();

use crate::{
    OperatorStorage,
    StorageOrder,
    OperatorDef,
    GradientMakerBase,
    CPUContext
};

/**
  | Applies a separate affine transformation
  | to each channel of the input. Useful
  | for replacing spatial batch norm with
  | its equivalent fixed transformation.
  |
  */
pub struct AffineChannelOp<T, Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage:      OperatorStorage,
    context:      Context,
    order:        StorageOrder,
    is_learnable: bool,
    phantom:      PhantomData<T>,
}

num_inputs!{AffineChannel, 3}

num_outputs!{AffineChannel, 1}

inputs!{AffineChannel, 
    0 => ("X", "Feature map input with order NCHW or NHWC."),
    1 => ("scale", "1D input of shape (C); the c-th element is the scale factor of the affine transformation for the c-th channel of the input."),
    2 => ("bias", "1D input of shape (C); the c-th element is the bias of the affine transformation for the c-th channel of the input.")
}

outputs!{AffineChannel, 
    0 => ("Y", "Output with the same order of Input.")
}

allow_inplace!{AffineChannel, vec![(0, 0)]}

impl<T,Context> AffineChannelOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
          order_(StringToStorageOrder(
                  this->template GetSingleArgument<std::string>("order", "NCHW"))),
                  OP_SINGLE_ARG(bool, "is_learnable", is_learnable_, false) 

                      CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return order_ == StorageOrder::NCHW ? RunOnDeviceWithOrderNCHW()
                                            : RunOnDeviceWithOrderNHWC();
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
        const auto& X = Input(0);
        const auto& scale = Input(1);
        const auto& bias = Input(2);

        if (is_learnable_) {
          CAFFE_ENFORCE(
              !IsInputOutputAlias(0, 0),
              "In-place affine_channel_op is not supported when "
              "is_learnable = true.");
        }
        const int N = X.dim32(0);
        const int C = X.dim32(1);
        const int HxW = X.numel() / (N * C);
        auto* Y = Output(0, X.sizes(), at::dtype<T>());
        math::AffineChannel<T, Context, StorageOrder::NCHW>(
            N,
            C,
            HxW,
            X.template data<T>(),
            scale.template data<T>(),
            bias.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_orderNHWC(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
        const auto& scale = Input(1);
        const auto& bias = Input(2);

        if (is_learnable_) {
          CAFFE_ENFORCE(
              !IsInputOutputAlias(0, 0),
              "In-place affine_channel_op is not supported when "
              "is_learnable = true.");
        }
        const int ndim = X.dim();
        const int N = X.dim32(0);
        const int C = X.dim32(ndim - 1);
        const int HxW = X.numel() / (N * C);
        auto* Y = Output(0, X.sizes(), at::dtype<T>());
        math::AffineChannel<T, Context, StorageOrder::NHWC>(
            N,
            C,
            HxW,
            X.template data<T>(),
            scale.template data<T>(),
            bias.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}

///-------------------------------------------------------

pub struct AffineChannelGradientOp<T, Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    order:        StorageOrder,
    is_learnable: bool,
    phantom: PhantomData<T>,
}

num_inputs!{AffineChannelGradient,    (2,3)}

num_outputs!{AffineChannelGradient,   (1,3)}

allow_inplace!{AffineChannelGradient, vec![(0, 0)]}

impl<T,Context> AffineChannelGradientOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
          order_(StringToStorageOrder(
                  this->template GetSingleArgument<std::string>("order", "NCHW"))),
                  OP_SINGLE_ARG(bool, "is_learnable", is_learnable_, false) 

                      CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return order_ == StorageOrder::NCHW ? RunOnDeviceWithOrderNCHW()
                                            : RunOnDeviceWithOrderNHWC();
        */
    }
}

impl AffineChannelGradientOp<f32, CPUContext> {

    #[inline] pub fn run_on_device_with_order_nchw(&mut self) -> bool {
        
        todo!();
        /*
            const auto& dY = Input(0);
      const auto& scale = is_learnable_ ? Input(2) : Input(1);

      auto* dX = Output(0, dY.sizes(), at::dtype<float>());
      const int N = dY.dim32(0);
      const int C = dY.dim32(1);
      const int HxW = dY.numel() / (N * C);
      const float* dY_data = dY.data<float>();
      const float* scale_data = scale.data<float>();
      const std::array<int, 3> X_dims = {N, C, HxW};
      const std::array<int, 3> scale_dims = {1, C, 1};
      math::Mul<float, CPUContext>(
          3,
          X_dims.data(),
          3,
          scale_dims.data(),
          dY_data,
          scale_data,
          dX->template mutable_data<float>(),
          &context_);
      if (is_learnable_) {
        const auto& X = Input(1);
        const float* X_data = X.data<float>();

        auto* dscale = Output(1, scale.sizes(), at::dtype<float>());
        auto* dbias = Output(2, scale.sizes(), at::dtype<float>());
        AffineChannelScaleBiasBackwardNCHW<float>(
            N,
            C,
            HxW,
            dY_data,
            X_data,
            dscale->template mutable_data<float>(),
            dbias->template mutable_data<float>());
      }
      return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_order_nhwc(&mut self) -> bool {
        
        todo!();
        /*
            const auto& dY = Input(0);
      const auto& scale = is_learnable_ ? Input(2) : Input(1);

      auto* dX = Output(0, dY.sizes(), at::dtype<float>());
      const int ndim = dY.dim();
      const int C = dY.dim32(ndim - 1);
      const int rows = dY.numel() / C;
      const int cols = C;
      const float* dY_data = dY.data<float>();
      const float* scale_data = scale.data<float>();
      math::RowwiseMul<float, CPUContext>(
          rows,
          cols,
          dY_data,
          scale_data,
          dX->template mutable_data<float>(),
          &context_);
      if (is_learnable_) {
        const auto& X = Input(1);
        const float* X_data = X.data<float>();
        const int N = X.dim32(0);
        const int HxW = rows / N;

        auto* dscale = Output(1, scale.sizes(), at::dtype<float>());
        auto* dbias = Output(2, scale.sizes(), at::dtype<float>());
        AffineChannelScaleBiasBackwardNHWC<float>(
            N,
            C,
            HxW,
            dY_data,
            X_data,
            dscale->template mutable_data<float>(),
            dbias->template mutable_data<float>());
      }
      return true;
        */
    }
}

#[inline] pub fn affine_channel_scale_bias_backwardNCHW<T>(
    n:        i32,
    c:        i32,
    hxW:      i32,
    dY:       *const T,
    x:        *const T,
    dscale:   *mut T,
    dbias:    *mut T) 
{
    todo!();
    /*
        const T* dY_ptr = dY;
      const T* X_ptr = X;
      const int stride = C * HxW;
      EigenVectorArrayMap<T> dscale_arr(dscale, C);
      EigenVectorArrayMap<T> dbias_arr(dbias, C);
      dscale_arr.setZero();
      dbias_arr.setZero();
      for (int i = 0; i < N; ++i) {
        ConstEigenArrayMap<T> dY_arr(dY_ptr, HxW, C);
        ConstEigenArrayMap<T> X_arr(X_ptr, HxW, C);
        dscale_arr += (dY_arr * X_arr).colwise().sum();
        dbias_arr += dY_arr.colwise().sum();
        dY_ptr += stride;
        X_ptr += stride;
      }
    */
}

#[inline] pub fn affine_channel_scale_bias_backwardNHWC<T>(
    n:         i32,
    c:         i32,
    hxW:       i32,
    dY:        *const T,
    x:         *const T,
    dscale:    *mut T,
    dbias:     *mut T) 
{
    todo!();
    /*
        ConstEigenArrayMap<T> dY_arr(dY, C, N * HxW);
      ConstEigenArrayMap<T> X_arr(X, C, N * HxW);
      EigenVectorMap<T>(dscale, C) = (dY_arr * X_arr).rowwise().sum();
      EigenVectorMap<T>(dbias, C) = dY_arr.rowwise().sum();
    */
}

register_cpu_operator!{AffineChannel,          AffineChannelOp<f32, CPUContext>}

register_cpu_operator!{AffineChannelGradient,  AffineChannelGradientOp<f32, CPUContext>}

pub struct GetAffineChannelGradient;

impl GetGradientDefs for GetAffineChannelGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            ArgumentHelper arg_helper(def_);
        const bool is_learnable =
            arg_helper.GetSingleArgument("is_learnable", false);
        if (is_learnable) {
          return SingleGradientDef(
              "AffineChannelGradient",
              "",
              std::vector<std::string>{GO(0), I(0), I(1)},
              std::vector<std::string>{GI(0), GI(1), GI(2)});
        } else {
          return SingleGradientDef(
              "AffineChannelGradient",
              "",
              std::vector<std::string>{GO(0), I(1)},
              std::vector<std::string>{GI(0)});
        }
        */
    }
}

register_gradient!{AffineChannel, GetAffineChannelGradient}
