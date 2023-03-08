crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct NormalizeGradientOp<T,Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{NormalizeGradient, 2}

num_outputs!{NormalizeGradient, 1}

args!{NormalizeGradient, 
    0 => ("axis", "axis to normalize")
}

input_tags!{
    NormalizeGradientOp {
        Input,
        GradOut
    }
}

output_tags!{
    NormalizeGradientOp {
        GradIn
    }
}

impl<T,Context> NormalizeGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
        const T kEps_ = EPS;
            const auto& x = Input(0);
        const auto& gOut = Input(GRAD_OUT);

        auto* gIn = Output(GRAD_IN, gOut.sizes(), at::dtype<T>());

        const auto* xData = x.template data<T>();
        const auto* gOutData = gOut.template data<T>();
        auto* gInData = gIn->template mutable_data<T>();

        const auto canonical_axis = x.canonical_axis_index(
            this->template GetSingleArgument<int>("axis", -1));
        const int m = x.dim32(canonical_axis);
        const int n = x.numel() / m;
        const int sf = x.size_from_dim(canonical_axis + 1);
        DoNormalize(xData, gOutData, gInData, m, n, sf);
        return true;
        */
    }
    
    #[inline] pub fn do_normalize(&mut self, 
        x_data:     *const T,
        g_out_data: *const T,
        g_in_data:  *mut T,
        m:          i32,
        n:          i32,
        sf:         i32)  
    {
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
            ConstStridedVec gOutVec(gOutData + base, 1, m, InnerStride(sf));

            auto row_sum = xVec.dot(gOutVec);
            auto row_norm = xVec.template lpNorm<2>();
            row_norm = std::max(row_norm, kEps_);
            auto row_norm_3 = pow(row_norm, 3);
            StridedVec gInVec(gInData + base, 1, m, InnerStride(sf));
            gInVec = (gOutVec / row_norm) - ((xVec / row_norm_3) * row_sum);
          }
        */
    }
}
