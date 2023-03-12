crate::ix!();

/**
  | UnaryFunctorWithDefaultCtor is a functor that
  | can be used as the functor of an
  | UnaryElementwiseWithArgsOp.
  |
  | It simply forwards the operator() call into
  | another functor that doesn't accept arguments
  | in its constructor.
  */
pub struct UnaryFunctorWithDefaultCtor<Functor> {
    functor: Functor,
}

impl<Functor> UnaryFunctorWithDefaultCtor<Functor> {

    #[inline] pub fn invoke<TIn, TOut, Context>(
        size: i32, 
        x: *const TIn, 
        y: *mut TOut, 
        context: *mut Context) -> bool 
    {
        todo!();
        /*
            return functor(size, X, Y, context);
        */
    }
}

/**
  | UnaryElementwiseOp is a wrapper around
  | UnaryElementwiseWithArgsOp, with the
  | difference that it takes a functor with
  | default constructor, e.g. that does not need
  | to take into consideration any arguments
  | during operator creation.
  */
pub type UnaryElementwiseOp<InputTypes, Context, Functor, OutputTypeMap> = 
UnaryElementwiseWithArgsOp<InputTypes, Context, UnaryFunctorWithDefaultCtor<Functor>, OutputTypeMap>;
