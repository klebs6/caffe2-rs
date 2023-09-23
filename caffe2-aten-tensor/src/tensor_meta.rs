crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/TensorMeta.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/TensorMeta.cpp]

/**
  | Use this to define the prototype for a meta
  | function.  There are two versions; one that
  | takes one argument (just the operator name), or
  | FUNC2 variant that takes two arguments
  | (operator name and overload name).
  |
  | Example usage:
  |
  |    TORCH_META_FUNC2(add, Tensor) (
  |      const Tensor& self, const Tensor& other
  |    ) {
  |      ... compute sizes and options ...
  |      set_output(sizes, options);
  |    }
  |
  */
#[macro_export] macro_rules! torch_meta_func {
    ($name:ident) => {
        /*
                void structured_##name::meta
        */
    }
}

#[macro_export] macro_rules! torch_meta_func2 {
    ($name:ident, $overload:ident) => {
        /*
                void structured_##name##_##overload::meta
        */
    }
}

/**
  | Use this to define the prototype for an
  | implementation.  This takes only one argument,
  | which is the name of the dispatch key entry
  | you're implementing.
  |
  | Example usage:
  |
  |    TORCH_IMPL_FUNC(add_cpu) (
  |      Tensor& result, const Tensor& self, const Tensor& other
  |    ) {
  |      ... do the actual implementation ...
  |    }
  |
  */
#[macro_export] macro_rules! torch_impl_func {
    ($name:ident) => {
        /*
                void structured_##name::impl
        */
    }
}

pub trait MetaBaseInterface:
SetOutput
+ MaybeGetOutput {}

pub trait SetOutput {

    fn set_output(&mut self, 
        output_idx: i64,
        sizes:      &[i32],
        strides:    &[i32],
        options:    TensorOptions,
        names:      &[Dimname]);
}

pub trait MaybeGetOutput {

    fn maybe_get_output(&mut self, output_idx: i64) -> &Tensor;
}

/**
  | Base class for all structured kernel classes.
  | The set_output virtual method is varied
  | depending whether or not the operator is
  | functional/out/inplace, and could also be
  | specialized for CPU/CUDA/etc (although
  | presently it isn't).
  |
  | A notable subclass of this interface is
  | TensorIteratorBase.
  */
pub struct MetaBase {

}

impl MetaBase {
    
    pub fn set_output(&mut self, 
        sizes:   &[i32],
        options: TensorOptions)  {
        
        todo!();
        /*
            set_output(0, sizes, {}, options, {});
        */
    }
    
    pub fn set_output(&mut self, 
        output_idx: i64,
        sizes:      &[i32],
        options:    TensorOptions)  {
        
        todo!();
        /*
            set_output(output_idx, sizes, {}, options, {});
        */
    }

    /**
      | Returns a reference to an undefined
      | tensor if there is no presupplied output
      |
      */
    pub fn maybe_get_output(&mut self) -> &Tensor {
        
        todo!();
        /*
            return maybe_get_output(0);
        */
    }
}
