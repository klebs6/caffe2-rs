crate::ix!();

pub struct BinaryFunctorWithDefaultCtor<Functor> {
    functor: Functor,
}

impl<Functor> BinaryFunctorWithDefaultCtor<Functor> {

    #[inline] pub fn forward<TIn, TOut, Context>(
        &mut self,
        a_dims:   &Vec<i32>,
        b_dims:   &Vec<i32>,
        a_data:   *const TIn,
        b_data:   *const TIn,
        c_data:   *mut TOut,
        context:  *mut Context) -> bool 
    {
        todo!();
        /*
            return functor.Forward(A_dims, B_dims, A_data, B_data, C_data, context);
        */
    }

    #[inline] pub fn backward<TGrad, TIn, TOut, Context>(
        &mut self,
        a_dims:     &Vec<i32>,
        b_dims:     &Vec<i32>,
        dC_data:    *const TGrad,
        a_data:     *const TIn,
        b_data:     *const TIn,
        c_data:     *const TOut,
        dA_data:    *mut TGrad,
        dB_data:    *mut TGrad,
        context:    *mut Context) -> bool 
    {
        todo!();
        /*
           return functor.Backward(
                A_dims,
                B_dims,
                dC_data,
                A_data,
                B_data,
                C_data,
                dA_data,
                dB_data,
                context);
        */
    }
}
