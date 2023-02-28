crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cuda/CompositeRandomAccessor.h]

pub struct TupleInfoCPU {

}

impl TupleInfoCPU {
    
    pub fn tie<Types>(args: &mut Types) -> Auto {
    
        todo!();
        /*
            return thrust::tie(args...);
        */
    }
}

pub type CompositeRandomAccessorCPU<KeyAccessor,ValueAccessor> = CompositeRandomAccessor<KeyAccessor,ValueAccessor,TupleInfoCPU>;

pub fn swap<Values, References>(
        rh1: ReferencesHolder<Values,References>,
        rh2: ReferencesHolder<Values,References>)  {

    todo!();
        /*
            return thrust::swap(rh1.data(), rh2.data());
        */
}

lazy_static!{
    /*
    template <int N, typename Values, typename References>
    auto get(references_holder<Values, References> rh) -> decltype(thrust::get<N>(rh.data())) {
      return thrust::get<N>(rh.data());
    }
    */
}
