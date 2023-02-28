crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/templates/Operators.h]

/**
  | Extension writers: do you write wrapper
  | functions? Are you frustrated with resolving
  | overloads of operators? Are you frustrated with
  | dealing with pointer-to-methods and resolving
  | overloads of pointer-to-methods?? Look no
  | further, this is the utility for you.
  |
  | Given an operator schema: op.overload(...
  |
  | Use ATEN_FN2(op, overload) to get a *function*
  | version of the operator that is guaranteed to
  | not be overloaded. This means that you can
  | safely decltype(&ATEN_FN2(op, overload))
  | it. NB: the 2 means this macro takes 2 args.
  |
  | Given an operator schema without an overload
  | name: op(...
  |
  | Use ATEN_FN(op) to get an unambiguous
  | *function* version of the operator.
  |
  | There is some interesting behavior for out=
  | operations. ATEN_FN2(sin, out) gives a function
  | that is *faithful* to the schema; that is, the
  | order of arguments is exactly what it looks
  | like in the schema.
  */
#[macro_export] macro_rules! aten_fn2 {
    ($op_name:ident, $overload:ident) => {
        /*
                _ops::op_name##_##overload
        */
    }
}

#[macro_export] macro_rules! aten_fn {
    ($op_name:ident) => {
        /*
                _ops::op_name
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/templates/Operators.cpp]

/**
  | NB: We are forced to special case
  | requires_grad_. This is because all of the
  | auto-generated inplace method signatures in
  | TensorMethods.h are codegen'ed to return
  | Tensor&, but requires_grad_ has
  | a `manual_cpp_binding` with a different
  | signature that returns `const Tensor&`.
  |
  | Eventually, the plan is to kill Tensor& from
  | all C++ signatures and use const Tensor&. When
  | that happens, we can remove this special case
  | and just let the codegen handle it.
  */
pub fn requires_grad(
        self_:         &mut Tensor,
        requires_grad: bool) -> &mut Tensor {
    
    todo!();
        /*
            self.requires_grad_(requires_grad);
      return self;
        */
}
