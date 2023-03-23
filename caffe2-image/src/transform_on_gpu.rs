crate::ix!();

pub fn transform_on_gpu<T_IN, T_OUT, Context>(
    x:         &Tensor, 
    Y:         *mut Tensor, 
    mean:      &Tensor, 
    std:       &Tensor, 
    context:   *mut Context) -> bool 
{
    todo!();
}

