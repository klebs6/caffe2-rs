crate::ix!();


/**
 | @brief Creates a quantized tensor of the given
 | dimension.
 |
 | Note that the actual data allocation is not
 | going to be carried out until the first time
 |  mutable_data() is called.
 |
 | The underlying storage of the quantized tensor
 | interleaves elements by bit depth.
 |
 | Labeled memory for tensor of size 6, precision 3
 |
 |   [ E1[0] E2[0] E3[0] E4[0] E5[0] E6[0] ] // Least significant Bits
 |   [ E1[1] E2[1] E3[1] E4[1] E5[1] E6[1] ]
 |   [ E1[2] E2[2] E3[2] E4[2] E5[2] E6[2] ]
 |
 | In the case of sign bits (see enable_sign
 | argument), an extra bit per element is added:
 |
 | Labeled memory for tensor of size 6, precision
 | 3, sign bit enabled
 |
 |   [ E1[0] E2[0] E3[0] E4[0] E5[0] E6[0] ]
 |   [ E1[1] E2[1] E3[1] E4[1] E5[1] E6[1] ]
 |   [ E1[2] E2[2] E3[2] E4[2] E5[2] E6[2] ]
 |   [ E1[s] E2[s] E3[s] E4[s] E5[s] E6[s] ]
 |   Where 's' is 1 if E is negative
 |
 | The reason for this layout is the ability to
 | efficiently multiply many low precision integers
 |  as a sum of popcnt(A & B) * 1 << bit.
 |
 | Explained here:
 | https://arxiv.org/abs/1606.06160
 */
#[derive(Default)]
pub struct QTensor<Context> {

    dims: Vec<i32>,
    size: usize,

    /// Precision in bits.
    precision_: u8, //= CHAR_BIT;

    /// Bit alignment.
    alignment_: u8, // = CHAR_BIT;

    /// Allocated data.
    data_ptr: DataPtr,

    /// value = scale_ * (x + bias_)
    scale:  f64,
    bias:   f64,

    signed: bool,//false

    /// Capacity in bits.
    capacity: usize,//0

    phantom: PhantomData<Context>,
}

impl<Context> QTensor<Context> {
    
    #[inline] pub fn set_bit_at_index(
        &mut self, 
        bit: u8,
        index: usize,
        value: bool)  
    {
        todo!();
        /*
            // Get the mutable data at bit depth `bit`.
        unsigned char* d = mutable_data();

        CAFFE_ENFORCE(
            bit < precision_ + signed_,
            "Attempted to a set a bit that is not allocated.");
        CAFFE_ENFORCE(bit * aligned_size() < capacity_);

        auto idx = (aligned_size() * bit) / CHAR_BIT;
        d = &d[idx];

        idx = index / CHAR_BIT;
        auto shift = CHAR_BIT - (index % CHAR_BIT) - 1;

        if (value) {
          d[idx] |= 1 << shift;
        } else {
          d[idx] &= ~(1 << shift);
        }
        */
    }
    
    #[inline] pub fn get_bit_at_index(
        &self, 
        bit: u8,
        index: usize) -> bool 
    {
        todo!();
        /*
            // Get the data at bit depth `bit`
        const unsigned char* d = data();
        auto idx = (aligned_size() * bit) / CHAR_BIT;
        d = &d[idx];

        idx = index / CHAR_BIT;
        auto shift = CHAR_BIT - (index % CHAR_BIT) - 1;

        return d[idx] & (1 << shift);
        */
    }
    
    #[inline] pub fn set_precision(&mut self, precision: u8)  {
        
        todo!();
        /*
            precision_ = precision;
        data_ptr_.clear();
        */
    }
    
    #[inline] pub fn set_signed(&mut self, make_signed: Option<bool>)  {

        let make_signed: bool = make_signed.unwrap_or(true);

        todo!();
        /*
            signed_ = make_signed;
        data_ptr_.clear();
        */
    }
    
    #[inline] pub fn set_scale(&mut self, scale: f64)  {
        
        todo!();
        /*
            scale_ = scale;
        */
    }
    
    #[inline] pub fn set_bias(&mut self, bias: f64)  {
        
        todo!();
        /*
            bias_ = bias;
        */
    }
    
    #[inline] pub fn mutable_data(&mut self) -> *mut u8 {
        
        todo!();
        /*
            if (!data_ptr_) {
          data_ptr_ = Context::New(nbytes());
          capacity_ = nbytes() * CHAR_BIT;
        }
        CAFFE_ENFORCE(capacity_ == nbytes() * CHAR_BIT);
        return static_cast<unsigned char*>(data_ptr_.get());
        */
    }
    
    #[inline] pub fn data(&self) -> *const u8 {
        
        todo!();
        /*
            return static_cast<unsigned char*>(data_ptr_.get());
        */
    }
    
    #[inline] pub fn size(&self) -> usize {
        
        todo!();
        /*
            return size_;
        */
    }
    
    #[inline] pub fn alignment(&self) -> u8 {
        
        todo!();
        /*
            return alignment_;
        */
    }
    
    #[inline] pub fn precision(&self) -> u8 {
        
        todo!();
        /*
            return precision_;
        */
    }
    
    #[inline] pub fn is_signed(&self) -> bool {
        
        todo!();
        /*
            return signed_;
        */
    }
    
    /**
      | Returns the number of dimensions of
      | the data.
      |
      */
    #[inline] pub fn ndim(&self) -> i32 {
        
        todo!();
        /*
            return dims_.size();
        */
    }
    
    #[inline] pub fn aligned_size(&self) -> usize {
        
        todo!();
        /*
            return alignment_ * ((size_ + alignment_ - 1) / alignment_);
        */
    }
    
    #[inline] pub fn nbytes(&self) -> usize {
        
        todo!();
        /*
            return (aligned_size() * (precision_ + signed_)) / CHAR_BIT;
        */
    }
    
    #[inline] pub fn scale(&self) -> f64 {
        
        todo!();
        /*
            return scale_;
        */
    }
    
    #[inline] pub fn bias(&self) -> f64 {
        
        todo!();
        /*
            return bias_;
        */
    }
    
    /**
      | Returns the i-th dimension of the qtensor
      | in int.
      |
      */
    #[inline] pub fn dim32(&self, i: i32) -> i32 {

        todo!();
        /*
            DCHECK_LT(i, dims_.size()) << "Exceeding ndim limit " << dims_.size();
        DCHECK_GE(i, 0) << "Cannot have negative index";
        CAFFE_ENFORCE_LT(dims_[i], int::max);
        return static_cast<int>(dims_[i]);
        */
    }
    
    /**
      | Returns the 'canonical' version of
      | a (usually) user-specified axis, allowing
      | for negative indexing (e.g., -1 for
      | the last axis).
      | 
      | -----------
      | @param axis_index
      | 
      | the axis index.
      | 
      | If 0 <= index < ndim(), return index.
      | 
      | If -ndim <= index <= -1, return (ndim()
      | - (-index)), e.g., the last axis index
      | (ndim() - 1) if index == -1, the second
      | to last if index == -2, etc.
      | 
      | Dies on out of range index.
      |
      */
    #[inline] pub fn canonical_axis_index(&self, axis_index: i32) -> i32 {
        
        todo!();
        /*
            CAFFE_ENFORCE_GE(axis_index, -ndim());
        CAFFE_ENFORCE_LT(axis_index, ndim());
        if (axis_index < 0) {
          return axis_index + ndim();
        }
        return axis_index;
        */
    }
    
    /**
      | Return product of all dimensions starting
      | from K.
      |
      */
    #[inline] pub fn size_from_dim(&self, k: i32) -> i64 {
        
        todo!();
        /*
            int64_t r = 1;
        for (int i = k; i < dims_.size(); ++i) {
          r *= dims_[i];
        }
        return r;
        */
    }
    
    /**
      | Product of all dims up to.
      |
      */
    #[inline] pub fn size_to_dim(&self, k: i32) -> i64 {
        
        todo!();
        /*
            CAFFE_ENFORCE(k < dims_.size());
        int64_t r = 1;
        for (int i = 0; i < k; ++i) {
          r *= dims_[i];
        }
        return r;
        */
    }

    /// TODO: changing at::ArrayRef<int> to at::ArrayRef<int64_t>?
    pub fn new(dims: &[i32], precision: u8, signbit: bool) -> Self {
        todo!();
        /*
        : precision_(precision), signed_(signbit) 
            Resize(dims);
        */
    }

    pub fn resize(&mut self, dim_source: &[i32]) {
        todo!();
        /*
        if (dims_ != dim_source) {
          const auto source_size = c10::multiply_integers(dim_source);
          if ((source_size * (precision_ + signed_)) > capacity_) {
            data_ptr_.clear();
            capacity_ = 0;
          }
          dims_ = dim_source.vec();
          size_ = source_size;
        }
        */
    }

    #[inline] pub fn sizes<'a>(&'a mut self) -> &'a [i32] {
        &self.dims
    }

    //TODO: deprecate?
    #[inline] pub fn dims<'a>(&'a mut self) -> &'a [i32] {
        &self.dims
    }
}

caffe_known_type!{QTensor<CPUContext>}
