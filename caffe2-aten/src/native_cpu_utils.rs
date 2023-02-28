crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/utils.h]

#[inline] pub fn data_index_init<T, Args>(
        offset: T,
        x:      &mut T,
        X:      &T,
        args:   Args) -> T {

    /*
    // base case
    #[inline] pub fn data_index_init<T>(offset: T) -> T {

        todo!();
            /*
                return offset;
            */
    }
    */

    todo!();
        /*
            offset = data_index_init(offset, forward<Args>(args)...);
      x = offset % X;
      return offset / X;
        */
}

#[inline] pub fn data_index_step<T, Args>(
        x:    &mut T,
        X:    &T,
        args: Args) -> bool {

    #[inline] fn data_index_step_base_case() -> bool {
        
        todo!();
            /*
                return true;
            */
    }

    todo!();
        /*
            if (data_index_step(forward<Args>(args)...)) {
        x = ((x + 1) == X) ? 0 : (x + 1);
        return x == 0;
      }
      return false;
        */
}

pub fn ceil_log2<T>(x: &T) -> T {

    todo!();
        /*
            if (x <= 2) {
        return 1;
      }
      // Last set bit is floor(log2(x)), floor + 1 is ceil
      // except when x is an exact powers of 2, so subtract 1 first
      return static_cast<T>(llvm::findLastSet(static_cast<u64>(x) - 1)) + 1;
        */
}
