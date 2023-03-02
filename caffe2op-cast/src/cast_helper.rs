crate::ix!();

pub struct CastHelper<DstType, SrcType> {
    phantomA: PhantomData<DstType>,
    phantomB: PhantomData<SrcType>,
}

impl<DstType,SrcType> CastHelper<DstType, SrcType> {
    
    #[inline] pub fn call(data: SrcType) -> DstType {
        
        todo!();
        /*
            return static_cast<DstType>(data);
        */
    }
}
