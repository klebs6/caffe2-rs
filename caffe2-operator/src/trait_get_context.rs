crate::ix!();

pub trait GetContext<Context> {

    #[inline] fn get_context(&self) -> *const Context {
        
        todo!();
        /*
            return &context_;
        */
    }
}

pub trait GetContextMut<Context> {

    #[inline] fn get_context_mut(&mut self) -> *mut Context {
        
        todo!();
        /*
            return &context_;
        */
    }
}
