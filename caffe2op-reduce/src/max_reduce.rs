crate::ix!();

pub struct MaxReducer<Context> {
    phantom: PhantomData<Context>,
}

impl<Context> MaxReducer<Context> {

    #[inline] pub fn forward<T>(&self, 
        x_dims:  &Vec<i32>,
        y_dims:  &Vec<i32>,
        x_data:  *const T,
        y_data:  *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::ReduceMax<T, Context>(
            X_dims.size(),
            X_dims.data(),
            Y_dims.data(),
            T(1),
            X_data,
            Y_data,
            context);
        return true;
        */
    }
    
    #[inline] pub fn backward<T>(&self, 
        dY_dims: &Vec<i32>,
        dX_dims: &Vec<i32>,
        dY_data: *const T,
        x_data:  *const T,
        y_data:  *const T,
        dX_data: *mut T,
        context: *mut Context) -> bool {
    
        todo!();
        /*
        
        */
    }
}

