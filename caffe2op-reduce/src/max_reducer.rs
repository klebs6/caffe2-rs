crate::ix!();

pub struct MaxReducer<T> {

    base:          BaseReducer,
    context:       CPUContext,

    out:           *mut T,
    current_size:  i32,

    /*
       using FixedDispatch = FixedValues<1>;
       */
}

impl<T> MaxReducer<T> {
    
    pub fn new(
        meta:    &BaseReducerMeta,
        out:     *mut T,
        context: *mut CPUContext) -> Self {
    
        todo!();
        /*
            : out_(out), current_size_(0) 

        // add a wrapper in Context for it
        memset(out, 0, sizeof(T) * meta.block_size);
        */
    }
    
    #[inline] pub fn process<const FixedSize: i32>(&mut self, 
        meta:    &BaseReducerMeta,
        input:   *const T,
        offset:  i64,
        context: *mut CPUContext)  {

        todo!();
        /*
            CAFFE_ENFORCE(
            meta.first_dim,
            "MaxReducer implemented only for front dimensions reduction");
        if (current_size_ > 0) {
          EigenVectorMap<T> output_vec(out_, meta.block_size);
          output_vec =
              output_vec.cwiseMax(ConstEigenVectorMap<T>(in, meta.block_size));
        } else {
          memcpy(out_, in, sizeof(T) * meta.block_size);
        }
        ++current_size_;
        */
    }
}
