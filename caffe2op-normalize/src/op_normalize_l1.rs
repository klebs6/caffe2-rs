crate::ix!();

/**
  | Given a matrix, apply L1-normalization
  | along the specified axis.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct NormalizeL1Op<T,Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

register_cpu_operator!{NormalizeL1, NormalizeL1Op<f32, CPUContext>}

num_inputs!{NormalizeL1, 1}

num_outputs!{NormalizeL1, 1}

args!{NormalizeL1, 
    0 => ("axis", "axis to normalize")
}

impl<T,Context> NormalizeL1Op<T,Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& x = Input(0);

        const auto* xData = x.template data<T>();
        auto* y = Output(0, x.sizes(), at::dtype<T>());
        auto* yData = y->template mutable_data<T>();

        const auto canonical_axis = x.canonical_axis_index(
            this->template GetSingleArgument<int>("axis", -1));
        const int m = x.dim32(canonical_axis);
        const int n = x.numel() / m;
        const int sf = x.size_from_dim(canonical_axis + 1);
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
            using InnerStride = Eigen::InnerStride<Eigen::Dynamic>;
      using StridedVec =
          Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>, 0, InnerStride>;
      using ConstStridedVec =
          Eigen::Map<const Eigen::Matrix<T, 1, Eigen::Dynamic>, 0, InnerStride>;

      for (int i = 0; i < n; ++i) {
        auto base = (i / sf) * sf * m + (i % sf);
        ConstStridedVec xVec(xData + base, 1, m, InnerStride(sf));
        auto norm = xVec.template lpNorm<1>();
        if (norm != 0) {
          StridedVec yVec(yData + base, 1, m, InnerStride(sf));
          yVec = xVec / norm;
        }
      }
        */
    }
}
