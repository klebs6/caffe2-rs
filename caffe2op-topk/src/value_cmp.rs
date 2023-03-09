crate::ix!();

pub struct ValueCmp<T> { 
    phantom: PhantomData<T>,
}

impl<T> ValueCmp<T> {
    
    #[inline] pub fn invoke(&mut self, lhs: &(T,i64), rhs: &(T,i64)) -> bool {
        
        todo!();
        /*
            return (
            lhs.first > rhs.first ||
            (lhs.first == rhs.first && lhs.second < rhs.second));
        */
    }
}

pub struct ValueComp<T> { 
    phantom: PhantomData<T>,
}

impl<T> ValueComp<T> {
    
    #[inline] pub fn invoke(
        &self, 
        lhs: &(T,i64),
        rhs: &(T,i64)) -> bool 
    {
        todo!();
        /*
            return lhs.first > rhs.first ||
            (lhs.first == rhs.first && lhs.second < rhs.second);
        */
    }
}
