crate::ix!();

impl<T, Context> FloorOp<T, Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

        auto* Y = Output(0, X.sizes(), at::dtype<float>());

        const float* Xdata = X.template data<float>();
        float* Ydata = Y->template mutable_data<float>();
        for (int i = 0; i < X.numel(); ++i) {
          Ydata[i] = std::floor(Xdata[i]);
        }
        return true;
        */
    }
}
