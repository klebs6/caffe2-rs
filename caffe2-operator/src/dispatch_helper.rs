crate::ix!();

pub struct DispatchHelper<FixedValues, ExtraArgs> {
    phantomA: PhantomData<FixedValues>,
    phantomB: PhantomData<ExtraArgs>,
}

impl<FixedValues,ExtraArgs> 
DispatchHelper<FixedValues, ExtraArgs> {

    #[inline] pub fn call<Op>(op: *mut Op, size: i64) -> bool {
    
        todo!();
        /*
            return op->template DoRunWithValue<ExtraArgs..., -1>();
        */
    }
    
    #[inline] pub fn call_with_value<Op>(op: *mut Op, value: i32) -> bool {
    
        todo!();
        /*
            if (FirstVal == value) {
          return op->template DoRunWithValue<ExtraArgs..., FirstVal>();
        }
        return DispatchHelper<FixedValues<Values...>, ExtraArgs...>::template call<
            Op>(op, value);
        */
    }
}

/**
  | Helpers to implement runtime op
  | polymorphism. Often it's convenient to make an
  | op work on different input types (e.g. i32 vs
  | i64 indices) or special-case it for particular
  | input size (e.g. ScatterWeightedSum for block
  | size of 1 doesn't need to call Eigen).
  |
  | DispatchHelper provides compile-time generation
  | of nested "if" statements,
  | e.g. `DispatchHelper<FixedValues<1,
  | 4>>::call(this, block_size);` unrolls into:
  |
  | @code
  |   if (block_size == 1) {
  |     return DoRunWithValue<1>();
  |   } else if (block_size = 4) {
  |     return DoRunWithValue<4>();
  |   } else {
  |     return DoRunWithValue<-1>();
  |   }`
  | @endcode
  |
  | DoRunWithValue implementation can use template
  | arguments to do "if" statements or proxy to
  | functions in math.h which often provide fixed
  | size implementation.
  |
  | Similarly `TensorTypes<int32_t, int64_t>(this,
  | Input(0))` provides branching based on type of
  | the first input and calls DoRunWithType.
  |
  | Note, that the same instance of Op class is
  | used as the method, not class is templated. We
  | might consider adding static class-level
  | polymorphism later.
  |
  | Convenient macro USE_DISPATCH_HELPER is
  | provided for declaring friendship in case
  | DoRunWithValue or DoRunWithType are declared
  | non-public.
  */
#[macro_export] macro_rules! use_dispatch_helper {
    () => {
        todo!();
        /*
        
          template <typename FirstArg, typename... ExtraArgs> 
          friend struct DispatchHelper
        */
    }
}
