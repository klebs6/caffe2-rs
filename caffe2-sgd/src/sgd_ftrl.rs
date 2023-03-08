crate::ix!();

use crate::{
    Operator,
    OperatorStorage,
    CPUContext,
    OperatorDef,
    Workspace
};

pub struct FtrlParams<T> {
    alpha_inv:  T,
    beta:       T,
    lambda1:    T,
    lambda2:    T,
}

impl<T> FtrlParams<T> {

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

///------------------------
// TODO(dzhulgakov): implement GPU version if necessary
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FtrlOp<T,Context> {
    context: Context,
    params:  FtrlParams<T>,
}

impl<T,Context> Operator for FtrlOp<T,Context> {

}

register_cpu_operator!{Ftrl, FtrlOp<float, CPUContext>}

num_inputs!{Ftrl, (3,4)}

num_outputs!{Ftrl, 2}

allow_inplace!{Ftrl, vec![(0, 0), (1, 1)]}

should_not_do_gradient!{Ftrl}

impl<T,Context> FtrlOp<T,Context> {

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
    FtrlOp
    {
        Var,
        NZ,
        Grad,
        Alpha
    }
}

output_tags!{
    FtrlOp
    {
        OutputVar,
        OutputNZ
    }
}

///------------------------------------------
pub struct SparseFtrlOp<T> {
    params:  FtrlParams<T>,
}

impl<T> Operator for SparseFtrlOp<T> {

}

register_cpu_operator!{SparseFtrl, SparseFtrlOp<float>}

num_inputs!{SparseFtrl, (4,5)}

num_outputs!{SparseFtrl, 2}

enforce_inplace!{SparseFtrl, vec![(0, 0), (1, 1)]}

should_not_do_gradient!{SparseFtrl}

impl<T> SparseFtrlOp<T> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws), params_(this) 

        CAFFE_ENFORCE(
            !HasArgument("alpha") || ALPHA >= InputSize(),
            "Cannot specify alpha by both input and argument");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // run time learning rate override
        if (ALPHA < InputSize()) {
          CAFFE_ENFORCE_EQ(Input(ALPHA).numel(), 1, "alpha should be real-valued");
          params_.alphaInv = 1.0 / *(Input(ALPHA).template data<T>());
        }
        // Use run-time polymorphism
        auto& indices = Input(INDICES);
        if (indices.template IsType<int32_t>()) {
          DoRun<int32_t>();
        } else if (indices.template IsType<int64_t>()) {
          DoRun<int64_t>();
        } else {
          LOG(FATAL) << "Unsupported type of INDICES in SparseFtrlOp: "
                     << indices.dtype().name();
        }
        return true;
        */
    }
}

input_tags!{
    SparseFtrlOp
    {
        Var,
        NZ,
        Indices,
        Grad,
        Alpha
    }
}

output_tags!{
    SparseFtrlOp
    {
        OutputVar,
        OutputNZ
    }
}

///-----------------------------
#[inline] pub fn sgn<T>(x: T) -> T {

    todo!();
    /*
        return (x == 0 ? 0 : (x < 0 ? -1 : 1));
    */
}

#[inline] pub fn ftrl_compute<T>(
    w:      T,
    n:      T,
    z:      T,
    g:      T,
    nw:     &mut T,
    nn:     &mut T,
    nz:     &mut T,
    params: &FtrlParams<T>)  {

    todo!();
    /*
        auto new_n = n + g * g;
      auto sigma = (sqrt(new_n) - sqrt(n)) * params.alphaInv;
      nn = new_n;
      nz = z + g - sigma * w;
      // update the weight
      if (std::abs(nz) > params.lambda1) {
        nw = (params.lambda1 * sgn(nz) - nz) /
            ((params.beta + sqrt(new_n)) * params.alphaInv + params.lambda2);
      } else {
        nw = 0.0;
      }
    */
}

// TODO(dzhulgakov): implement SIMD-based version
#[inline] pub fn ftrl_update<Context, T>(
    n:       i32,
    w:       *const T,
    nz:      *const T,
    g:       *const T,
    new_w:   *mut T,
    new_nz:  *mut T,
    params:  &FtrlParams<T>,
    context: *mut Context)  {

    todo!();
    /*
        // TODO(cxj): use OMP when it is reliable
      // #pragma omp parallel for
      for (auto i = 0; i < N; ++i) {
        ftrl_compute(
            w[i],
            nz[i * 2],
            nz[i * 2 + 1],
            g[i],
            new_w[i],
            new_nz[i * 2],
            new_nz[i * 2 + 1],
            params);
      }
    */
}

impl<T,Context> FtrlOp<T, Context> {

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
      ftrl_update<Context>(
          Input(GRAD).numel(),
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

impl<T> SparseFtrlOp<T> {

    #[inline] pub fn do_run<SIndex>(&mut self)  {
    
        todo!();
        /*
            auto* var = Output(OUTPUT_VAR);
      auto* n_z = Output(OUTPUT_N_Z);
      auto& indices = Input(INDICES);
      auto& grad = Input(GRAD);
      CAFFE_ENFORCE_EQ(&Input(VAR), var, "In place operation is required");
      CAFFE_ENFORCE_EQ(&Input(N_Z), n_z, "In place operation is required");
      int64_t M = var->numel();
      int64_t N = var->size(0);
      int64_t block_size = M / N;
      int64_t K = indices.numel();
      DCHECK_EQ(M * 2, n_z->numel());
      DCHECK_EQ(grad.numel(), K * block_size);
      T* w = var->template mutable_data<T>();
      T* nz = n_z->template mutable_data<T>();
      const SIndex* idxs = indices.template data<SIndex>();
      const T* g = grad.template data<T>();

      // TODO(cxj): use OMP when it is reliable
      // #pragma omp parallel for
      for (int64_t i = 0; i < K; ++i) {
        SIndex idx = idxs[i];
        DCHECK(0 <= idx && idx < N) << "Index out of bounds: " << idx
                                    << ", range 0 to " << N;
        if (block_size == 1) {
          ftrl_compute(
              w[idx],
              nz[idx * 2],
              nz[idx * 2 + 1],
              g[i],
              w[idx],
              nz[idx * 2],
              nz[idx * 2 + 1],
              params_);
        } else {
          int64_t x = block_size * idx;
          ftrl_update(
              block_size,
              w + x,
              nz + x * 2,
              g + i * block_size,
              w + x,
              nz + x * 2,
              params_,
              &context_);
        }
      }
        */
    }
}

