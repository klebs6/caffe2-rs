crate::ix!();

/**
 | Embedding lookup with reduction.
 |
 | `input` of size data_size * (block_size + 8B)
 | `indices` of size index_size
 | `lengths` of size output_size
 | `weights` nullptr or array of size index_size
 | `out` of size output_size * block_size
 | sum(lengths[i]) == index_size
 |
 | Note that block_size should be the number of
 | quantized values per row in the data,
 | i.e. excluding the scale and bias. The total
 | (fused) block size is assumed to be this
 | block_size, plus 4 bytes for scale and 4 bytes
 | for bias.
 |
 | Behavior is roughly equivalent to pseudocode:
 |
 | pos = 0
 | fused_block_size = block_size + 8B // quantized values and scale and bias
 | for (i = 0..output_size-1)
 |   for (k = 0..block_size-1)
 |     out[i*block_size + k] = 0
 |   for (j = 0..lengths[i]-1)
 |     for (k = 0..block_size-1)
 |       out[i*block_size + k] += input[indices[pos]*(fused_block_size) + k] *
 |           (weights ? weights[IS_WEIGHT_POSITIONAL ? j : pos] : 1.0)
 |     pos += 1
 |   if (normalize_weights && lengths[i] > 0)
 |     for (k = 0..block_size-1)
 |       out[i*block_size + k] /= lengths[i]
 |
 |
 |  bool IS_WEIGHT_POSITIONAL = false>
 |  const float* weights, // optional, can be null for non-weighted sum
 */
#[inline] pub fn fused_8bit_rowwise_embedding_lookup<IndexType, InType, OutType, const IS_WEIGHT_POSITIONAL: bool>(
    block_size:           i64,
    output_size:          i64,
    index_size:           i64,
    data_size:            i64,
    input:                *const InType,
    indices:              *const IndexType,
    lengths:              *const i32,
    weights:              *const f32,
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
 */
#[inline] pub fn fused_8bit_rowwise_embedding_lookup_generic_slow<IndexType, InType, OutType, const IS_WEIGHT_POSITIONAL: bool>(
    block_size:           i64,
    output_size:          i64,
    index_size:           i64,
    data_size:            i64,
    input:                *const InType,
    indices:              *const IndexType,
    lengths:              *const i32,
    weights:              *const f32,
    normalize_by_lengths: bool,
    out:                  *mut OutType) -> bool {

    todo!();
    /*
        // block_size is the number of elements and fused_block_size is the size of
      // an entire row, including scale and bias.
      const auto scale_bias_offset = 8 / sizeof(InType);
      const int64_t fused_block_size = block_size + scale_bias_offset;
      int64_t current = 0;
      for (const auto m : c10::irange(output_size)) {
        memset(out, 0, sizeof(OutType) * block_size);
        if (current + lengths[m] > index_size) {
          return false;
        }
        for (int i = 0; i < lengths[m]; ++i) {
          int64_t idx = indices[current];
          if (idx < 0 || idx >= data_size) {
            return false;
          }
    #ifdef __GNUC__
          if (current + 1 < index_size) {
            __builtin_prefetch(
                input + fused_block_size * indices[current + 1], 0, 1);
          }
    #endif // __GNUC__

          const float* scale_bias = reinterpret_cast<const float*>(
              input + fused_block_size * indices[current] + block_size);

          float weight = 1.0f;
          if (weights) {
            weight = weights[IS_WEIGHT_POSITIONAL ? i : current];
          }
          const float scale = weight * scale_bias[0];
          const float bias = weight * scale_bias[1];

          for (const auto j : c10::irange(block_size)) {
            out[j] += scale * input[fused_block_size * indices[current] + j] + bias;
          }

          ++current;
        }
        if (normalize_by_lengths && lengths[m]) {
          float scale = 1.f / lengths[m];
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
macro_rules! fused_8bit_rowwise_embedding_specialization {
    ($IndexType:ident, $OutType:ident) => {
        /*
        
          bool                                                                                   
              Fused8BitRowwiseEmbeddingLookup_##IndexType##_uint8_t_##OutType##_false__base(     
                  const int64_t block_size,                                                      
                  const int64_t output_size,                                                     
                  const int64_t index_size,                                                      
                  const int64_t data_size,                                                       
                  const uint8_t* input,                                                          
                  const IndexType* indices,                                                      
                  const int* lengths,                                                            
                  const float* weights,                                                          
                  bool normalize_by_lengths,                                                     
                  OutType* out) {                                                                
            return Fused8BitRowwiseEmbeddingLookupGenericSlow<                                   
                IndexType,                                                                       
                uint8_t,                                                                         
                OutType,                                                                         
                false>(                                                                          
                block_size,                                                                      
                output_size,                                                                     
                index_size,                                                                      
                data_size,                                                                       
                input,                                                                           
                indices,                                                                         
                lengths,                                                                         
                weights,                                                                         
                normalize_by_lengths,                                                            
                out);                                                                            
          }                                                                                      
          decltype(                                                                              
              Fused8BitRowwiseEmbeddingLookup_##IndexType##_uint8_t_##OutType##_false__base)     
              Fused8BitRowwiseEmbeddingLookup_##IndexType##_uint8_t_##OutType##_false__avx2_fma; 
          bool Fused8BitRowwiseEmbeddingLookup_##IndexType##_uint8_t_##OutType(                  
              const int64_t block_size,                                                          
              const int64_t output_size,                                                         
              const int64_t index_size,                                                          
              const int64_t data_size,                                                           
              const uint8_t* input,                                                              
              const IndexType* indices,                                                          
              const int* lengths,                                                                
              const float* weights,                                                              
              bool normalize_by_lengths,                                                         
              OutType* out) {                                                                    
            const int32_t one = 1;                                                               
            CAFFE_ENFORCE_EQ(                                                                    
                reinterpret_cast<const uint8_t*>(&one)[0],                                       
                1,                                                                               
                "Fused8BitRowwiseEmbeddingLookup is not supported on this platform");            
            AVX2_FMA_DO(                                                                         
                Fused8BitRowwiseEmbeddingLookup_##IndexType##_uint8_t_##OutType##_false,         
                block_size,                                                                      
                output_size,                                                                     
                index_size,                                                                      
                data_size,                                                                       
                input,                                                                           
                indices,                                                                         
                lengths,                                                                         
                weights,                                                                         
                normalize_by_lengths,                                                            
                out);                                                                            
            BASE_DO(                                                                             
                Fused8BitRowwiseEmbeddingLookup_##IndexType##_uint8_t_##OutType##_false,         
                block_size,                                                                      
                output_size,                                                                     
                index_size,                                                                      
                data_size,                                                                       
                input,                                                                           
                indices,                                                                         
                lengths,                                                                         
                weights,                                                                         
                normalize_by_lengths,                                                            
                out);                                                                            
          }                                                                                      
          template <>                                                                            
          void Fused8BitRowwiseEmbeddingLookup<IndexType, uint8_t, OutType, false>(              
              const int64_t block_size,                                                          
              const int64_t output_size,                                                         
              const int64_t index_size,                                                          
              const int64_t data_size,                                                           
              const uint8_t* input,                                                              
              const IndexType* indices,                                                          
              const int* lengths,                                                                
              const float* weights,                                                              
              bool normalize_by_lengths,                                                         
              OutType* out) {                                                                    
            bool success =                                                                       
                Fused8BitRowwiseEmbeddingLookup_##IndexType##_uint8_t_##OutType(                 
                    block_size,                                                                  
                    output_size,                                                                 
                    index_size,                                                                  
                    data_size,                                                                   
                    input,                                                                       
                    indices,                                                                     
                    lengths,                                                                     
                    weights,                                                                     
                    normalize_by_lengths,                                                        
                    out);                                                                        
            if (success) {                                                                       
              return;                                                                            
            }                                                                                    
            int64_t current = 0;                                                                 
            for (int m = 0; m < output_size; ++m) {                                              
              for (int i = 0; i < lengths[m]; ++i) {                                             
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

fused_8bit_rowwise_embedding_specialization!{i32, f32}
fused_8bit_rowwise_embedding_specialization!{i64, f32}
