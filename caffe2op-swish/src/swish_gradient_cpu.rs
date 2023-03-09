crate::ix!();

register_cpu_operator!{
    SwishGradient, 
    SwishGradientOp<CPUContext>
}

impl<CPUContext> SwishGradientOp<CPUContext> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        
        todo!();
        /*
            auto& Xin = Input(X);
                auto& Yin = Input(Y);
                auto& DYin = Input(DY);

                CAFFE_ENFORCE_EQ(Xin.numel(), Yin.numel());
                CAFFE_ENFORCE_EQ(DYin.numel(), Yin.numel());
                auto* DXout = Output(DX, Yin.sizes(), at::dtype<float>());

                const float* Xdata = Xin.template data<float>();
                const float* Ydata = Yin.template data<float>();
                const float* dYdata = DYin.template data<float>();
                float* dXdata = DXout->template mutable_data<float>();

                EigenVectorArrayMap<float> dXvec(dXdata, DXout->numel());
                ConstEigenVectorArrayMap<float> Xvec(Xdata, Xin.numel());
                ConstEigenVectorArrayMap<float> Yvec(Ydata, Yin.numel());
                ConstEigenVectorArrayMap<float> dYvec(dYdata, DYin.numel());

                // dx = dy * (y + sigmoid(x)*(1-y))
                dXvec = dYvec * (Yvec + (T(1) / (T(1) + (-Xvec).exp())) * (T(1) - Yvec));
                return true;
        */
    }
}
