crate::ix!();

impl BooleanMaskOpGradient<CPUContext> {

    #[inline] pub fn do_run_with_type<T>() -> bool {
        todo!();
        /*
            const auto& mask = Input(0);
          const auto& dY = Input(1);
          auto* dX = Output(0);

          const int data_length_before_mask = mask.size(0);

          dX->Resize(data_length_before_mask);

          // TODO: we should support any type, not just float
          T* dXdata = dX->template mutable_data<T>();
          const T* dYdata = dY.template data<T>();
          const bool* mask_data = mask.template data<bool>();

          int ind = 0;

          for (int i = 0; i < data_length_before_mask; i++) {
            dXdata[i] = mask_data[i] ? dYdata[ind++] : 0;
          }

          return true;
        */
    }
}
