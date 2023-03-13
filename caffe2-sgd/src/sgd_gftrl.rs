crate::ix!();


pub struct GFtrlParams<T> {
    
    alpha_inv:  T,
    beta:       T,
    lambda1:    T,
    lambda2:    T,
}

impl<T> GFtrlParams<T> {

    pub fn new(op: *mut OperatorStorage) -> Self {
    
        todo!();
        /*
            : alphaInv(1.0 / op->GetSingleArgument<float>("alpha", 0.005f)),
            beta(op->GetSingleArgument<float>("beta", 1.0f)),
            lambda1(op->GetSingleArgument<float>("lambda1", 0.001f)),
            lambda2(op->GetSingleArgument<float>("lambda2", 0.001f))
        */
    }
}

///--------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GFtrlOp<T,Context> {
    context: Context,
    params:  GFtrlParams<T>,
}

impl<T,Context> Operator for GFtrlOp<T,Context> {
}

impl<T,Context> GFtrlOp<T,Context> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator(operator_def, ws), params_(this) 

        CAFFE_ENFORCE(
            !HasArgument("alpha") || ALPHA >= InputSize(),
            "Cannot specify alpha by both input and argument");
        */
    }
}

input_tags!{
    GFtrlOp
    {
        Var,
        NZ,
        Grad,
        Alpha
    }
}

output_tags!{
    GFtrlOp
    {
        OutputVar,
        OutputNZ
    }
}


/// Computes one coordinate
#[inline] pub fn gftrl_compute<T>(
    w:          &T,
    n:          &T,
    z:          &T,
    g:          &T,
    nw:         &mut T,
    nn:         &mut T,
    nz:         &mut T,
    z_norm:     &T,
    output_dim: i32,
    params:     &GFtrlParams<T>)  {
    
    todo!();
    /*
        auto new_n = n + g * g;
      auto sigma = (sqrt(new_n) - sqrt(n)) * params.alphaInv;
      nn = new_n;
      nz = z + g - sigma * w;
      // update the weight
      if (z_norm > params.lambda1 * std::sqrt(OutputDim)) {
        nw = nz * (params.lambda1 * std::sqrt(OutputDim) / z_norm - 1) /
            ((params.beta + sqrt(new_n)) * params.alphaInv + params.lambda2);
      } else {
        nw = 0.0;
      }
    */
}

/**
 | int OutputDim, // # of output nodes
 | int InputDim, // # of input features
 */
#[inline] pub fn gftrl_update<Context, T>(
    output_dim: i32,
    input_dim:  i32,
    w:          *const T,
    nz:         *const T,
    g:          *const T,
    new_w:      *mut T,
    new_nz:     *mut T,
    params:     &GFtrlParams<T>,
    context:    *mut Context)  {

    todo!();
    /*
        for (auto j = 0; j < InputDim; ++j) {
        T z_norm = 0.0;
        for (auto i = 0; i < OutputDim; ++i) {
          int idx = i * InputDim + j;
          auto new_n = nz[idx * 2] + g[idx] * g[idx];
          auto sigma = (sqrt(new_n) - sqrt(nz[idx * 2])) * params.alphaInv;
          auto new_z = nz[idx * 2 + 1] + g[idx] - sigma * w[idx];
          z_norm = z_norm + new_z * new_z;
        }

        z_norm = sqrt(z_norm);
        for (auto i = 0; i < OutputDim; ++i) {
          int idx = i * InputDim + j;
          gftrl_compute(
              w[idx],
              nz[idx * 2],
              nz[idx * 2 + 1],
              g[idx],
              new_w[idx],
              new_nz[idx * 2],
              new_nz[idx * 2 + 1],
              z_norm,
              OutputDim,
              params);
        }
      }
    */
}

impl<T,Context> GFtrlOp<T,Context> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // run time learning rate override
      if (ALPHA < InputSize()) {
        CAFFE_ENFORCE_EQ(Input(ALPHA).numel(), 1, "alpha should be real-valued");
        params_.alphaInv = 1.0 / *(Input(ALPHA).template data<T>());
      }

      CAFFE_ENFORCE_EQ(Input(GRAD).numel(), Input(VAR).numel());
      CAFFE_ENFORCE_EQ(Input(GRAD).numel() * 2, Input(N_Z).numel());
      Output(OUTPUT_VAR)->ResizeLike(Input(VAR));
      Output(OUTPUT_N_Z)->ResizeLike(Input(N_Z));
      gftrl_update<Context>(
          Input(GRAD).size(0), // # of output nodes
          Input(GRAD).numel() / Input(GRAD).size(0), // # of input features
          Input(VAR).template data<T>(),
          Input(N_Z).template data<T>(),
          Input(GRAD).template data<T>(),
          Output(OUTPUT_VAR)->template mutable_data<T>(),
          Output(OUTPUT_N_Z)->template mutable_data<T>(),
          params_,
          &context_);
      return true;
        */
    }
}

register_cpu_operator!{GFtrl, GFtrlOp<float, CPUContext>}

num_inputs!{GFtrl, (3,4)}

num_outputs!{GFtrl, 2}

allow_inplace!{GFtrl, vec![(0, 0)]}

should_not_do_gradient!{GFtrl}

