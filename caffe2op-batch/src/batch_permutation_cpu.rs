crate::ix!();

impl BatchPermutationOp<f32, CPUContext> {

    #[inline] pub fn run_on_deviceA(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
          auto& indices = Input(1);

          CAFFE_ENFORCE(indices.dim() == 1, "indices must be 1-d");
          CAFFE_ENFORCE(
              X.dim32(0) == indices.dim32(0),
              "X.dim32(0) must be equal to indices.dim32(0)",
              "(",
              X.dim32(0),
              " vs. ",
              indices.dim32(0),
              ")");

          auto* Y = Output(0, X.sizes(), at::dtype<float>());

          if (X.dim32(0) > 0) {
            batch_permutation_loop<true>(
                X.dim32(0),
                X.numel() / X.dim32(0),
                X.data<float>(),
                indices.data<int>(),
                Y->mutable_data<float>());
          }
          return true;
        */
    }

    #[inline] pub fn run_on_deviceB(&mut self) -> bool {
        
        todo!();
        /*
            auto& indices = Input(0);
          auto& dY = Input(1);

          auto* dX = Output(0, dY.sizes(), at::dtype<float>());

          if (dY.dim32(0) > 0) {
            batch_permutation_loop<false>(
                dY.dim32(0),
                dY.numel() / dY.dim32(0),
                dY.data<float>(),
                indices.data<int>(),
                dX->mutable_data<float>());
          }
          return true;
        */
    }
}

