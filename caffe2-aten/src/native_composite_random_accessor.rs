crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/CompositeRandomAccessor.h]

pub struct TupleInfoCPU {

}

impl TupleInfoCPU {
    
    pub fn tie<Types>(args: &mut Types) -> Auto {
    
        todo!();
        /*
            return tie(args...);
        */
    }
}


lazy_static!{
    /*
    template <typename KeyAccessor, typename ValueAccessor>
    using CompositeRandomAccessorCPU =
      CompositeRandomAccessor<KeyAccessor, ValueAccessor, TupleInfoCPU>;
    */
}


pub fn swap<Values, References>(
        rh1: ReferencesHolder<Values,References>,
        rh2: ReferencesHolder<Values,References>)  {

    todo!();
        /*
            return swap(rh1.data(), rh2.data());
        */
}

lazy_static!{
    /*
    template <int N, typename Values, typename References>
    auto get(references_holder<Values, References> rh) -> decltype(get<N>(rh.data())) {
      return get<N>(rh.data());
    }
    */
}

