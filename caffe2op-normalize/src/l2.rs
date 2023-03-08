crate::ix!();

/**
  | Given a matrix, apply L2-normalization
  | along the specified dimension.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct NormalizeOp<T,Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{Normalize, 1}

num_outputs!{Normalize, 1}

args!{Normalize, 
    0 => ("axis", "axis to normalize")
}

identical_type_and_shape!{Normalize}

pub const EPS: f32 = 1e-12;

register_cpu_operator!{Normalize, NormalizeOp<float, CPUContext>}

register_cpu_gradient_operator!{NormalizeGradient, NormalizeGradientOp<float, CPUContext>}

impl<T,Context> NormalizeOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& x = Input(0);

        const auto* xData = x.template data<T>();
        auto* y = Output(0, x.sizes(), at::dtype<T>());
        auto* yData = y->template mutable_data<T>();

        const auto canonical_axis = x.canonical_axis_index(
            this->template GetSingleArgument<int>("axis", -1));
        const int64_t m = x.dim(canonical_axis);
        const size_t n = x.numel() / m;
        const size_t sf = x.size_from_dim(canonical_axis + 1);
        DoNormalize(xData, yData, m, n, sf);
        return true;
        */
    }
    
    #[inline] pub fn do_normalize(&mut self, 
        x_data: *const T,
        y_data: *mut T,
        m:      i32,
        n:      i32,
        sf:     i32)  {
        
        todo!();
        /*
           const T kEps_ = EPS;

            using InnerStride = Eigen::InnerStride<Eigen::Dynamic>;
        using StridedVec =
            Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>, 0, InnerStride>;
        using ConstStridedVec =
            Eigen::Map<const Eigen::Matrix<T, 1, Eigen::Dynamic>, 0, InnerStride>;

        for (int i = 0; i < n; ++i) {
          auto base = (i / sf) * sf * m + (i % sf);
          ConstStridedVec xVec(xData + base, 1, m, InnerStride(sf));
          auto norm = xVec.template lpNorm<2>();
          norm = std::max(norm, kEps_);
          StridedVec yVec(yData + base, 1, m, InnerStride(sf));
          yVec = xVec / norm;
        }
        */
    }
}
