crate::ix!();

#[inline] pub fn decode<CodebookT, CodeT>(
    codebook:     &Tensor,
    codes:        &Tensor,
    decoded_grad: *const Tensor,
    output:       *mut Tensor,
    resize_only:  bool) 
{
    todo!();
    /*
        CAFFE_ENFORCE(codebook.IsType<CodebookT>());

      auto* cb_ptr = codebook.data<CodebookT>();
      int cb_size = codebook.numel();

      CAFFE_ENFORCE(codes.IsType<CodeT>());
      auto* code_ptr = codes.data<CodeT>();

      if (decoded_grad == nullptr) {
        // Forward pass: decode and store codebook values in output.
        output->ResizeLike(codes);
        auto* out_ptr = output->template mutable_data<CodebookT>();
        if (resizeOnly) {
          return;
        }

        int sz = output->numel();
        for (int i = 0; i < sz; i++) {
          DCHECK_LE(*code_ptr, cb_size);
          *out_ptr++ = cb_ptr[*code_ptr++];
        }
      } else {
        // Backward pass: decode and accumulate gradient w.r.t. codebook values.
        CAFFE_ENFORCE_EQ(codes.numel(), decoded_grad->numel());
        auto* gradient_ptr = decoded_grad->data<CodebookT>();
        auto* const gradient_end = gradient_ptr + decoded_grad->numel();

        CAFFE_ENFORCE_EQ(cb_size, output->numel());
        auto* out_ptr = output->template mutable_data<CodebookT>();
        while (gradient_ptr < gradient_end) {
          DCHECK_LE(*code_ptr, cb_size);
          out_ptr[*code_ptr++] += *gradient_ptr++;
        }
      }
    */
}

#[macro_export] macro_rules! REGISTER_DECODER {
    ($codebookType:ident, $codesType:ident) => {
        todo!();
        /*
        {                                                                    
            {TypeMeta::Id<codebookType>(), TypeMeta::Id<codesType>()},         
            [](const Tensor& codebook_,                                    
                const Tensor& codes_,                                       
                const Tensor* gradient_,                                    
                Tensor* outDecoded_,                                        
                bool resizeOnly_) {                                         
                Decode<codebookType, codesType>(                             
                    codebook_, codes_, gradient_, outDecoded_, resizeOnly_); 
            }                                                              
        }
        */
    }
}

#[inline] pub fn decode_general(
    codebook:     &Tensor,
    codes:        &Tensor,
    gradient:     *const Tensor,
    out_decoded:  *mut Tensor,
    resize_only:  bool)  
{
    todo!();
    /*
        const static std::map<
          std::pair<TypeIdentifier, TypeIdentifier>,
          std::function<void(
              const Tensor& codebook,
              const Tensor& codes,
              const Tensor* gradient,
              Tensor* outDecoded,
              bool resizeOnly)>>
          gDecoderMapper = {REGISTER_DECODER(float, uint8_t),
                            REGISTER_DECODER(float, uint16_t),
                            REGISTER_DECODER(float, int32_t)};

      gDecoderMapper.at({codebook.dtype().id(), codes.dtype().id()})(
          codebook, codes, gradient, outDecoded, resizeOnly);
    */
}
