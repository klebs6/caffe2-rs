crate::ix!();

///---------------------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct DotProductWithPaddingGradientOp<T, Context> {
    storage: OperatorStorage,
    context: Context,

    pad_value: f32,
    replicate: bool,

    phantom: PhantomData<T>,
}

num_inputs!{DotProductWithPaddingGradient, 3}

num_outputs!{DotProductWithPaddingGradient, 2}

register_gradient!{
    DotProductWithPadding, 
    GetDotProductWithPaddingGradient
}

register_cpu_operator!{
    DotProductWithPaddingGradient,
    DotProductWithPaddingGradientOp<float, CPUContext>
}

input_tags!{
    DotProductWithPaddingGradientOp {
        XIn,
        YIn,
        DerDotIn
    }
}

output_tags!{
    DotProductWithPaddingGradientOp {
        DerXOut,
        DerYOut
    }
}

impl<T,Context> DotProductWithPaddingGradientOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            pad_value_(this->template GetSingleArgument<float>("pad_value", 0.0)),
            replicate_(this->template GetSingleArgument<bool>("replicate", false))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(X_IN);
        auto& Y = Input(Y_IN);
        auto& dDot = Input(DER_DOT_IN);

        int N, D, DX, DY, restD;
        if (X.numel() > 0) {
          N = X.dim() > 0 ? X.dim32(0) : 1;
          DX = X.numel() / N;
          DY = Y.numel() / N;
        } else {
          N = 0;
          DX = 0;
          DY = 0;
        }
        CAFFE_ENFORCE(!replicate_ || DX % DY == 0 || DY % DX == 0);
        D = std::min(DX, DY);
        restD = std::max(DX, DY) - D;
        CAFFE_ENFORCE_EQ(X.dim(), Y.dim());
        CAFFE_ENFORCE_EQ(X.dim32(0), Y.dim32(0));
        CAFFE_ENFORCE_EQ(dDot.dim(), 1);
        CAFFE_ENFORCE_EQ(dDot.dim32(0), N);
        auto* dX = Output(DER_X_OUT, X.sizes(), at::dtype<T>());
        auto* dY = Output(DER_Y_OUT, Y.sizes(), at::dtype<T>());

        const auto* X_data = X.template data<T>();
        const auto* Y_data = Y.template data<T>();
        const auto* dDot_data = dDot.template data<T>();
        auto* dX_data = dX->template mutable_data<T>();
        auto* dY_data = dY->template mutable_data<T>();
        for (int i = 0; i < N; ++i) { // TODO: multithreading
          auto offsetX = i * DX;
          auto offsetY = i * DY;
          if (replicate_) {
            // L_ for longer vector and S_ for shorter vector
            const T *L_data, *S_data;
            T *dL_data, *dS_data;
            int DL, DS;
            if (DX > DY) {
              L_data = X_data + offsetX;
              S_data = Y_data + offsetY;
              dL_data = dX_data + offsetX;
              dS_data = dY_data + offsetY;
              DL = DX;
              DS = DY;
            } else {
              L_data = Y_data + offsetY;
              S_data = X_data + offsetX;
              dL_data = dY_data + offsetY;
              dS_data = dX_data + offsetX;
              DL = DY;
              DS = DX;
            }

            // TODO: get rid of temp memory use
            std::vector<T> tmp_data(DS);
            math::Set<T, Context>(DS, 0.0, dS_data, &context_);
            for (int j = 0; j < DL / DS; j++) {
              math::Scale<T, T, Context>(
                  DS, dDot_data[i], S_data, dL_data + j * DS, &context_);
              math::Scale<T, T, Context>(
                  DS, dDot_data[i], L_data + j * DS, tmp_data.data(), &context_);
              math::Axpy<float, T, Context>(
                  DS, 1.0, tmp_data.data(), dS_data, &context_);
            }
          } else {
            math::Scale<T, T, Context>(
                D, dDot_data[i], X_data + offsetX, dY_data + offsetY, &context_);
            math::Scale<T, T, Context>(
                D, dDot_data[i], Y_data + offsetY, dX_data + offsetX, &context_);
          }

          if (!replicate_ && DX != DY) {
            T* rest_data;
            if (DX > DY) {
              rest_data = dX_data + offsetX + D;
            } else {
              rest_data = dY_data + offsetY + D;
            }
            auto pad_gradient = dDot_data[i] * pad_value_;
            math::Set<T, Context>(restD, pad_gradient, rest_data, &context_);
          }
        }

        return true;
        */
    }
}

pub struct GetDotProductWithPaddingGradient;

impl GetGradientDefs for GetDotProductWithPaddingGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            float pad_value = 0;
        bool replicate = false;
        if (ArgumentHelper::HasArgument(Def(), "pad_value")) {
          pad_value = GetArgument(Def(), "pad_value").f();
        }
        if (ArgumentHelper::HasArgument(Def(), "replicate")) {
          replicate = GetArgument(Def(), "replicate").i();
        }

        const auto dot_arg =
            vector<Argument>{MakeArgument<float>("pad_value", pad_value),
                             MakeArgument<bool>("replicate", replicate)};

        return SingleGradientDef(
            "DotProductWithPaddingGradient",
            "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(0), GI(1)},
            dot_arg);
        */
    }
}
