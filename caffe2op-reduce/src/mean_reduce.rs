crate::ix!();

pub struct MeanReducer<Context> {
    phantom: PhantomData<Context>,
}

impl<Context> MeanReducer<Context> {

    #[inline] pub fn forward<T>(&self, 
        x_dims:  &Vec<i32>,
        y_dims:  &Vec<i32>,
        x_data:  *const T,
        y_data:  *mut T,
        context: *mut Context) -> bool {

        todo!();
        /*
            math::ReduceMean<T, Context>(
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
            const int dY_size = std::accumulate(
            dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
        const int dX_size = std::accumulate(
            dX_dims.cbegin(), dX_dims.cend(), 1, std::multiplies<int>());
        math::Broadcast(
            dY_dims.size(),
            dY_dims.data(),
            dX_dims.size(),
            dX_dims.data(),
            static_cast<T>(dY_size) / static_cast<T>(dX_size),
            dY_data,
            dX_data,
            context);
        return true;
        */
    }
}
