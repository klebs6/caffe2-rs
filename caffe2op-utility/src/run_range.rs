crate::ix!();

impl<Context> RangeOp<Context> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int32_t, int64_t, float, double>>::call(
            this, Input(0));
        */
    }
    
    #[inline] pub fn read_scalar_input<T>(&mut self, index: i32) -> T {
        todo!();
        /*
            if (std::is_same<Context, TensorCPU>::value) {
          return Input(index).template data<T>()[0];
        } else {
          local_.CopyFrom(Input(index));
          return local_.template data<T>()[0];
        }
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            T stop = 0;
        T start = 0;
        T step = 1;

        for (int i = 0; i < InputSize(); ++i) {
          CAFFE_ENFORCE_EQ(
              Input(i).numel(), 1, "All inputs must be scalar/1D tensor.");
        }

        switch (InputSize()) {
          case 1:
            stop = readScalarInput<T>(0);
            break;
          case 2:
            start = readScalarInput<T>(0);
            stop = readScalarInput<T>(1);
            break;
          case 3:
            step = readScalarInput<T>(2);
            start = readScalarInput<T>(0);
            stop = readScalarInput<T>(1);
            break;
        }
        CAFFE_ENFORCE_NE(step, 0, "Step size cannot be 0.");
        int length;
        auto diff = stop - start;
        if (std::is_integral<T>::value) {
          // Avoid casting to and from floats in case it introduces rounding and
          // avoid mod because the compiler doesn't strip unused code until later.
          length = diff / step;
          if (length * step < diff) {
            length += 1;
          }
        } else {
          length = static_cast<int>(ceil(diff / step));
        }

        // Match numpy's behavior here.
        if (length <= 0) {
          Output(0, {0}, at::dtype<T>());
          return true;
        } else {
          auto* output = Output(0, {length}, at::dtype<T>());
          return DoRunOnDevice<T>(start, step, output);
        }
        */
    }
}

