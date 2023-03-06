crate::ix!();

impl GroupNormGradientOp<f32, CPUContext> {

    /**
      | Math:
      | let: s = gamma * rsig
      | let: b = beta - mu * gamma * rsig
      | then: Y = s * X + b
      */
    #[inline] pub fn run_on_device_with_orderNCHW(
        &mut self, 
        n:             i32,
        g:             i32,
        k:             i32,
        hxW:           i32,
        dY_data:       *const f32,
        x_data:        *const f32,
        mu_data:       *const f32,
        rsig_data:     *const f32,
        gamma_data:    *const f32,
        dX_data:       *mut f32,
        dgamma_data:   *mut f32,
        dbeta_data:    *mut f32) -> bool 
    {
        todo!();
        /*
            const int C = G * K;
      ReinitializeTensor(&ds_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&db_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&dY_scale_, {N, C}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&X_scale_, {N, G}, at::dtype<float>().device(CPU));
      ReinitializeTensor(&bias_, {N, G}, at::dtype<float>().device(CPU));
      float* ds_data = ds_.mutable_data<float>();
      float* db_data = db_.mutable_data<float>();
      float* dY_scale_data = dY_scale_.mutable_data<float>();
      float* X_scale_data = X_scale_.mutable_data<float>();
      float* bias_data = bias_.mutable_data<float>();
      ComputeInternalGradients<float, StorageOrder::NCHW>(
          N, C, HxW, dY_data, X_data, ds_data, db_data);
      ComputeGradientFusedParams<float>(
          N,
          G,
          K,
          HxW,
          ds_data,
          db_data,
          mu_data,
          rsig_data,
          gamma_data,
          dY_scale_data,
          X_scale_data,
          bias_data);
      GroupNormBackward<float, StorageOrder::NCHW>(
          N,
          G,
          K,
          HxW,
          dY_scale_data,
          dY_data,
          X_scale_data,
          X_data,
          bias_data,
          dX_data);
      GammaBetaBackward<float>(
          N, G, K, ds_data, db_data, mu_data, rsig_data, dgamma_data, dbeta_data);
      return true;
        */
    }

    #[inline] pub fn run_on_device_with_order_nhwc<T, Context>(
        &mut self,
        n:            i32,
        g:            i32,
        k:            i32,
        hxW:          i32,
        dY_data:      *const T,
        x_data:       *const T,
        mu_data:      *const T,
        rsig_data:    *const T,
        gamma_data:   *const T,
        dX_data:      *mut T,
        dgamma_data:  *mut T,
        dbeta_data:   *mut T) -> bool 
    {
        todo!();
        /*
            const int C = G * K;
          ReinitializeTensor(&ds_, {N, C}, at::dtype<float>().device(CPU));
          ReinitializeTensor(&db_, {N, C}, at::dtype<float>().device(CPU));
          ReinitializeTensor(&dY_scale_, {N, C}, at::dtype<float>().device(CPU));
          ReinitializeTensor(&X_scale_, {N, G}, at::dtype<float>().device(CPU));
          ReinitializeTensor(&bias_, {N, G}, at::dtype<float>().device(CPU));
          float* ds_data = ds_.mutable_data<float>();
          float* db_data = db_.mutable_data<float>();
          float* dY_scale_data = dY_scale_.mutable_data<float>();
          float* X_scale_data = X_scale_.mutable_data<float>();
          float* bias_data = bias_.mutable_data<float>();
          ComputeInternalGradients<float, StorageOrder::NHWC>(
              N, C, HxW, dY_data, X_data, ds_data, db_data);
          ComputeGradientFusedParams<float>(
              N,
              G,
              K,
              HxW,
              ds_data,
              db_data,
              mu_data,
              rsig_data,
              gamma_data,
              dY_scale_data,
              X_scale_data,
              bias_data);
          GroupNormBackward<float, StorageOrder::NHWC>(
              N,
              G,
              K,
              HxW,
              dY_scale_data,
              dY_data,
              X_scale_data,
              X_data,
              bias_data,
              dX_data);
          GammaBetaBackward<float>(
              N, G, K, ds_data, db_data, mu_data, rsig_data, dgamma_data, dbeta_data);
          return true;
        */
    }
}
