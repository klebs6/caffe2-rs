crate::ix!();

/**
 | Embedding lookup with reduction.
 |
 | `input` of size data_size * block_size
 | `indices` of size index_size
 | `offsets` of size output_size
 | `weights` nullptr or array of size index_size
 | `out` of size output_size * block_size
 |
 | Behavior is roughly equivalent to pseudocode:
 |
 | pos = 0
 | for (i = 0..output_size-1)
 |   for (k = 0..block_size-1)
 |     out[i*block_size + k] = 0
 |   start_offset = offsets[i]
 |   end_offset = offsets[i+1]
 |   length = end_offset - start_offset
 |   for (j = start_offset..end_offset-1)
 |     for (k = 0..block_size-1)
 |       out[i*block_size + k] += input[indices[pos]*block_size + k] *
 |           (weights ? weights[IS_WEIGHT_POSITIONAL ? j - start_offset : pos] : 1.0)
 |     pos += 1
 |   if (normalize_weights && length > 0)
 |     for (k = 0..block_size-1)
 |       out[i*block_size + k] /= length
 |
 | TODO: make this API also take "offsets" rather
 | than "lengths" to match the API for PyTorch's
 | EmbeddingBag
 |
 |  bool IS_WEIGHT_POSITIONAL = false>
 |  const float* weights, // optional, can be null for non-weighted sum
 |  const float* scale_bias, // optional scale & bias params for uint8 input
 */
#[inline] pub fn embedding_lookup_idx<IndexType, InType, OutType, const IS_WEIGHT_POSITIONAL: bool>(
    block_size:           i64,
    output_size:          i64,
    index_size:           i64,
    data_size:            i64,
    input:                *const InType,
    indices:              *const IndexType,
    offsets:              *const IndexType,
    weights:              *const f32,
    scale_bias:           *const f32,
    normalize_by_lengths: bool,
    out:                  *mut OutType)  {

    todo!();
    /*
    
    */
}

/**
 | Base implementation does runtime dispatch for
 | each segment of reduction
 |
 | @return false if there is an out-of-bound error
 |
 |  bool IS_WEIGHT_POSITIONAL = false>
 |  const float* weights, // optional, can be null for sum reducer
 |  const float* scale_bias, // optional scale & bias params for uint8 input
 */
#[inline] pub fn embedding_lookup_generic_slow_idx<IndexType, InType, OutType, const IS_WEIGHT_POSITIONAL: bool>(
    block_size:           i64,
    output_size:          i64,
    index_size:           i64,
    data_size:            i64,
    input:                *const InType,
    indices:              *const IndexType,
    offsets:              *const IndexType,
    weights:              *const f32,
    scale_bias:           *const f32,
    normalize_by_lengths: bool,
    out:                  *mut OutType) -> bool {

    todo!();
    /*
        int64_t current = 0;
      for (const auto m : c10::irange(output_size)) {
        memset(out, 0, sizeof(OutType) * block_size);
        if (current != offsets[m] - offsets[0]) {
          return false;
        }
        int64_t start_offset = offsets[m];
        int64_t end_offset = offsets[m + 1];
        int64_t length = end_offset - start_offset;
        for (const auto i : c10::irange(start_offset, end_offset)) {
          int64_t idx = indices[current];
          if (idx < 0 || idx >= data_size) {
            return false;
          }
    #ifdef __GNUC__
          if (current + 1 < index_size) {
            __builtin_prefetch(input + block_size * indices[current + 1], 0, 1);
          }
    #endif // __GNUC__

          float w = 1.f, b = 0.f;
          if (weights) {
            w = weights[IS_WEIGHT_POSITIONAL ? i - start_offset : current];
          }
          if (scale_bias) {
            b = w * scale_bias[2 * indices[current] + 1];
            w = w * scale_bias[2 * indices[current]];
          }

          for (const auto j : c10::irange(block_size)) {
            out[j] += w * input[block_size * indices[current] + j] + b;
          }

          ++current;
        }
        if (normalize_by_lengths && length) {
          float scale = 1.f / length;
          for (const auto j : c10::irange(block_size)) {
            out[j] *= scale;
          }
        }
        out += block_size;
      }
      return current == index_size;
    */
}


/// Proxy back to generic implementation
#[macro_export] macro_rules! embedding_idx_specialization {
    ($a:ty, $b:ty, $c:ty, $d:ty, $e:expr) => {
        /*
                (                                                                 
            IndexType, InTypeName, InType, OutType, IS_WEIGHT_POSITIONAL)                                     
          bool                                                                                                
              EmbeddingLookupIdx_##IndexType##_##InTypeName##_##OutType##_##IS_WEIGHT_POSITIONAL##__base(     
                  const int64_t block_size,                                                                   
                  const int64_t output_size,                                                                  
                  const int64_t index_size,                                                                   
                  const int64_t data_size,                                                                    
                  const InType* input,                                                                        
                  const IndexType* indices,                                                                   
                  const IndexType* offsets,                                                                     
                  const float* weights,                                                                       
                  const float* scale_bias,                                                                    
                  bool normalize_by_lengths,                                                                  
                  OutType* out) {                                                                             
            return EmbeddingLookupGenericSlowIdx<                                                             
                IndexType,                                                                                    
                InType,                                                                                       
                OutType,                                                                                      
                IS_WEIGHT_POSITIONAL>(                                                                        
                block_size,                                                                                   
                output_size,                                                                                  
                index_size,                                                                                   
                data_size,                                                                                    
                input,                                                                                        
                indices,                                                                                      
                offsets,                                                                                      
                weights,                                                                                      
                scale_bias,                                                                                   
                normalize_by_lengths,                                                                         
                out);                                                                                         
          }                                                                                                   
          decltype(                                                                                           
              EmbeddingLookupIdx_##IndexType##_##InTypeName##_##OutType##_##IS_WEIGHT_POSITIONAL##__base)     
              EmbeddingLookupIdx_##IndexType##_##InTypeName##_##OutType##_##IS_WEIGHT_POSITIONAL##__avx2_fma; 
          bool                                                                                                
              EmbeddingLookupIdx_##IndexType##_##InTypeName##_##OutType##_##IS_WEIGHT_POSITIONAL(             
                  const int64_t block_size,                                                                   
                  const int64_t output_size,                                                                  
                  const int64_t index_size,                                                                   
                  const int64_t data_size,                                                                    
                  const InType* input,                                                                        
                  const IndexType* indices,                                                                   
                  const IndexType* offsets,                                                                     
                  const float* weights,                                                                       
                  const float* scale_bias,                                                                    
                  bool normalize_by_lengths,                                                                  
                  OutType* out) {                                                                             
            if (is_same<InType, uint8_t>::value) {                                                       
              CAFFE_ENFORCE(scale_bias != nullptr, "scale_bias must not be nullptr");                         
            } else {                                                                                          
              CAFFE_ENFORCE(scale_bias == nullptr, "scale_bias must be nullptr");                             
            }                                                                                                 
            AVX2_FMA_DO(                                                                                      
                EmbeddingLookupIdx_##IndexType##_##InTypeName##_##OutType##_##IS_WEIGHT_POSITIONAL,           
                block_size,                                                                                   
                output_size,                                                                                  
                index_size,                                                                                   
                data_size,                                                                                    
                input,                                                                                        
                indices,                                                                                      
                offsets,                                                                                      
                weights,                                                                                      
                scale_bias,                                                                                   
                normalize_by_lengths,                                                                         
                out);                                                                                         
            BASE_DO(                                                                                          
                EmbeddingLookupIdx_##IndexType##_##InTypeName##_##OutType##_##IS_WEIGHT_POSITIONAL,           
                block_size,                                                                                   
                output_size,                                                                                  
                index_size,                                                                                   
                data_size,                                                                                    
                input,                                                                                        
                indices,                                                                                      
                offsets,                                                                                      
                weights,                                                                                      
                scale_bias,                                                                                   
                normalize_by_lengths,                                                                         
                out);                                                                                         
          }                                                                                                   
          template <>                                                                                         
          void EmbeddingLookupIdx<IndexType, InType, OutType, IS_WEIGHT_POSITIONAL>(                          
              const int64_t block_size,                                                                       
              const int64_t output_size,                                                                      
              const int64_t index_size,                                                                       
              const int64_t data_size,                                                                        
              const InType* input,                                                                            
              const IndexType* indices,                                                                       
              const IndexType* offsets,                                                                         
              const float* weights,                                                                           
              const float* scale_bias,                                                                        
              bool normalize_by_lengths,                                                                      
              OutType* out) {                                                                                 
            bool success =                                                                                    
                EmbeddingLookupIdx_##IndexType##_##InTypeName##_##OutType##_##IS_WEIGHT_POSITIONAL(           
                    block_size,                                                                               
                    output_size,                                                                              
                    index_size,                                                                               
                    data_size,                                                                                
                    input,                                                                                    
                    indices,                                                                                  
                    offsets,                                                                                  
                    weights,                                                                                  
                    scale_bias,                                                                               
                    normalize_by_lengths,                                                                     
                    out);                                                                                     
            if (success) {                                                                                    
              return;                                                                                         
            }                                                                                                 
            int64_t current = 0;                                                                              
            for (int m = 0; m < output_size; ++m) {                                                           
              for (int64_t i = offsets[m]; i < offsets[m + 1]; ++i) {                                         
                CAFFE_ENFORCE_LT(current, index_size);                                                        
                IndexType idx = indices[current];                                                             
                CAFFE_ENFORCE(                                                                                
                    0 <= idx && idx < data_size,                                                              
                    "Index ",                                                                                 
                    current,                                                                                  
                    " is out of bounds: ",                                                                    
                    idx,                                                                                      
                    ", range 0 to ",                                                                          
                    data_size);                                                                               
                ++current;                                                                                    
              }                                                                                               
            }                                                                                                 
            CAFFE_ENFORCE_EQ(                                                                                 
                current,                                                                                      
                index_size,                                                                                   
                "Your input seems to be incorrect: the sum of lengths values should be "                      
                "the size of the indices tensor, but it appears not.");                                       
          }
        */
    }
}


embedding_idx_specialization!{i32, f32, f32, f32, false}
embedding_idx_specialization!{i64, f32, f32, f32, false}
embedding_idx_specialization!{i32, f16, f16, f32, false}
embedding_idx_specialization!{i64, f16, f16, f32, false}
embedding_idx_specialization!{i32, u8, u8, f32, false}
embedding_idx_specialization!{i64, u8, u8, f32, false}

embedding_idx_specialization!{i32, f32, f32, f32, true}
embedding_idx_specialization!{i64, f32, f32, f32, true}
embedding_idx_specialization!{i32, f16, f16, f32, true}
embedding_idx_specialization!{i64, f16, f16, f32, true}
embedding_idx_specialization!{i32, u8, u8, f32, true}
embedding_idx_specialization!{i64, u8, u8, f32, true}
