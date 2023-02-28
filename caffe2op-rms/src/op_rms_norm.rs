crate::ix!();

use crate::{
    GradientMakerBase,
    OperatorDef,
    CPUContext,
    Tensor,
    OperatorStorage,
};

///-----------------------------------
/// RMSNorm op.
/// https://openreview.net/pdf?id=SygkZ3MTJE
pub struct RMSNormOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    axis: i32,
    eps:  f32,
}

num_inputs!{RMSNorm, 3}

num_outputs!{RMSNorm, 2}

inputs!{RMSNorm, 
    0 => ("input", "Input tensor which layer normalization will be applied to"),
    1 => ("gamma", "scale tensor for elementwise_affine, the shape should be the same as the dimensions of X begin from axis"),
    2 => ("beta",  "bias tensor for elementwise_affine, the shape should be the same as the dimensions of X begin from axis")
}

outputs!{RMSNorm, 
    0 => ("output","Normalized values"),
    1 => ("rrms",  "Reciprocal of root mean square for each feature vector")
}

args!{RMSNorm, 
    0 => ("axis",    "(int) default to 1; Describes axis of the inputs. Defaults to one because the 0th axis most likely describes the batch size"),
    1 => ("epsilon", "(float) default to 0.001. Small value to be added to the stdev when dividing out by that value. This prevents division by zero.")
}

tensor_inference_function!{RMSNorm, /* [](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      std::vector<TensorShape> out(2);
      const auto input_dims_long = GetDimsVector(in[0]);
      const std::vector<int> input_dims(
          input_dims_long.cbegin(), input_dims_long.cend());
      out[0] = CreateTensorShape(input_dims, in[0].data_type());
      ArgumentHelper helper(def);
      const int axis = helper.GetSingleArgument<int32_t>("axis", 1);
      const int canonical_axis =
          canonical_axis_index_(axis, in[0].dims().size());
      const std::vector<int> rms_dims(
          input_dims.cbegin(), input_dims.cbegin() + canonical_axis);
      out[1] = CreateTensorShape(rms_dims, in[0].data_type());
      return out;
    } */
}

impl<Context> RMSNormOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "axis", axis_, 1),
            OP_SINGLE_ARG(float, "eps", eps_, 0.0f)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
        */
    }
}

impl RMSNormOp<CPUContext> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(0);
          const auto& gamma = Input(1);
          const auto& beta = Input(2);
          auto* Y = Output(0, X.sizes(), at::dtype<T>());
          CAFFE_ENFORCE_GE(X.dim(), 2, "RMSNorm requires input dim >= 2.");
          const int canonical_axis = X.canonical_axis_index(axis_);
          const std::vector<int64_t> rms_dims(
              X.sizes().cbegin(), X.sizes().cbegin() + canonical_axis);
          auto* rrms = Output(1, rms_dims, at::dtype<T>());
          const int64_t M = X.size_to_dim(canonical_axis);
          const int64_t N = X.size_from_dim(canonical_axis);
          CAFFE_ENFORCE_EQ(gamma.numel(), N);
          CAFFE_ENFORCE_EQ(beta.numel(), N);

          const T* X_data = X.template data<T>();
          const T* gamma_data = gamma.template data<T>();
          const T* beta_data = beta.template data<T>();
          T* Y_data = Y->template data<T>();
          T* rrms_data = rrms->template data<T>();

          ConstEigenArrayMap<T> X_arr(X_data, N, M);
          ConstEigenVectorArrayMap<T> gamma_arr(gamma_data, N);
          ConstEigenVectorArrayMap<T> beta_arr(beta_data, N);
          EigenArrayMap<T> Y_arr(Y_data, N, M);
          at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
              const T rrms_val =
                  T(1) / std::sqrt(X_arr.col(i).square().mean() + static_cast<T>(eps_));
              Y_arr.col(i) = rrms_val * X_arr.col(i) * gamma_arr + beta_arr;
              rrms_data[i] = rrms_val;
            }
          });

          return true;
        */
    }
}

///-----------------------------------
pub struct RMSNormGradientOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    axis: i32,
    c2:   Tensor,
}

num_inputs!{RMSNormGradient, 4}

num_outputs!{RMSNormGradient, 3}

impl<Context> RMSNormGradientOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "axis", axis_, 1)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& dY = Input(0);
            const auto& X = Input(1);
            const auto& gamma = Input(2);
            const auto& rrms = Input(3);
            const int canonical_axis = X.canonical_axis_index(axis_);
            const int64_t M = X.size_to_dim(canonical_axis);
            const int64_t N = X.size_from_dim(canonical_axis);
            auto* dX = Output(0, X.sizes(), at::dtype<T>());
            auto* dgamma = Output(1, gamma.sizes(), at::dtype<T>());
            auto* dbeta = Output(2, gamma.sizes(), at::dtype<T>());
            const T* dY_data = dY.template data<T>();
            const T* X_data = X.template data<T>();
            const T* gamma_data = gamma.template data<T>();
            const T* rrms_data = rrms.template data<T>();
            T* dX_data = dX->template mutable_data<T>();
            T* dgamma_data = dgamma->template mutable_data<T>();
            T* dbeta_data = dbeta->template mutable_data<T>();

            if (M == 0) {
              math::Set<T, Context>(N, T(0), dgamma_data, &context_);
              math::Set<T, Context>(N, T(0), dbeta_data, &context_);
              return true;
            }

            RMSNormBackward<T>(M, N, dY_data, X_data, gamma_data, rrms_data, dX_data);
            GammaBetaBackward<T>(
                M, N, dY_data, X_data, rrms_data, dgamma_data, dbeta_data);

            return true;
        */
    }
}

impl RMSNormGradientOp<CPUContext> {

    #[inline] pub fn rms_norm_backward<T>(
        &mut self,
        m:      i64,
        n:      i64,
        dY:     *const T,
        x:      *const T,
        gamma:  *const T,
        rrms:   *const T,
        dX:     *mut T) 
    {
        todo!();
        /*
            ConstEigenArrayMap<T> dY_arr(dY, N, M);
          ConstEigenArrayMap<T> X_arr(X, N, M);
          ConstEigenVectorArrayMap<T> gamma_arr(gamma, N);
          EigenArrayMap<T> dX_arr(dX, N, M);
          const T scale = T(1) / static_cast<T>(N);
          at::parallel_for(0, M, 1, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
              const T ds = (dY_arr.col(i) * X_arr.col(i) * gamma_arr).sum();
              const T c1 = rrms[i];
              const T c2 = -scale * ds * math::utils::Cube<T>(rrms[i]);
              dX_arr.col(i) = c1 * dY_arr.col(i) * gamma_arr + c2 * X_arr.col(i);
            }
          });
        */
    }

    #[inline] pub fn gamma_beta_backward<T>(
        &mut self,
        m:        i64,
        n:        i64,
        dY:       *const T,
        x:        *const T,
        rrms:     *const T,
        dgamma:   *mut T,
        dbeta:    *mut T) 
    {
        todo!();
        /*
            math::Set<T, CPUContext>(N, T(0), dgamma, &context_);
          math::Set<T, CPUContext>(N, T(0), dbeta, &context_);
          ConstEigenArrayMap<T> dY_arr(dY, N, M);
          ConstEigenArrayMap<T> X_arr(X, N, M);
          EigenVectorArrayMap<T> dgamma_arr(dgamma, N);
          EigenVectorArrayMap<T> dbeta_arr(dbeta, N);
          for (int64_t i = 0; i < M; ++i) {
            dgamma_arr += dY_arr.col(i) * X_arr.col(i) * rrms[i];
            dbeta_arr += dY_arr.col(i);
          }
        */
    }
}

register_cpu_operator!{RMSNorm,         RMSNormOp<CPUContext>}

register_cpu_operator!{RMSNormGradient, RMSNormGradientOp<CPUContext>}

pub struct GetRMSNormGradient<'a> {

    base: GradientMakerStorage<'a>,
}

impl<'a> GetGradientDefs for GetRMSNormGradient<'a> {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "RMSNormGradient",
            "",
            std::vector<std::string>{GO(0), I(0), I(1), O(1)},
            std::vector<std::string>{GI(0), GI(1), GI(2)});
        */
    }
}

register_gradient!{RMSNorm, GetRMSNormGradient}
