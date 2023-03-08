crate::ix!();

#[USE_OPERATOR_FUNCTIONS(CPUContext)]
pub struct NormalizePlanarYUVOp {
    storage: OperatorStorage,
    context: CPUContext,
}

register_cpu_operator!{NormalizePlanarYUV, NormalizePlanarYUVOp}

num_inputs!{NormalizePlanarYUV, 3}

num_outputs!{NormalizePlanarYUV, 1}

allow_inplace!{NormalizePlanarYUV, vec![(0, 0)]}

impl NormalizePlanarYUVOp {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
        const auto& M = Input(1); // mean
        const auto& S = Input(2); // standard deviation

        auto* Z = Output(0, X.sizes(), at::dtype<float>());

        CAFFE_ENFORCE(X.sizes().size() == 4);

        const auto N = X.dim32(0);
        auto C = X.size(1);
        const auto H = X.size(2);
        const auto W = X.size(3);
        CAFFE_ENFORCE(C == M.size(1));
        CAFFE_ENFORCE(C == S.size(1));
        const auto* Xdata = X.data<float>();
        auto* Zdata = Z->template mutable_data<float>();

        int offset = H * W;
        for (auto n = 0; n < N; n++) { // realistically N will always be 1
          int batch_offset = n * C * offset;
          for (auto c = 0; c < C; c++) {
            ConstEigenVectorMap<float> channel_s(
                &Xdata[batch_offset + (c * offset)], offset);
            EigenVectorMap<float> channel_d(
                &Zdata[batch_offset + (c * offset)], offset);
            channel_d = channel_s.array() - M.data<float>()[c];
            channel_d = channel_d.array() / S.data<float>()[c];
          }
        }
        return true;
        */
    }
}
