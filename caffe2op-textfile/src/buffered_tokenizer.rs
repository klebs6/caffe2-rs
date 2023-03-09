crate::ix!();

pub struct BufferedTokenizer {
    provider:     *mut dyn StringProvider,
    tokenizer:    Tokenizer,
    tokenized:    TokenizedString,
    token_index:  i32,
    num_passes:   i32,
    pass:         i32, // {0};
}

impl BufferedTokenizer {
    
    #[inline] pub fn end_delim(&self) -> i32 {
        
        todo!();
        /*
            if (tokenIndex_ + 1 < tokenized_.tokens().size()) {
          return tokenized_.tokens()[tokenIndex_ + 1].startDelimId;
        }
        return tokenized_.lastDelim();
        */
    }
    
    #[inline] pub fn next(&mut self, token: &mut Token) -> bool {
        
        todo!();
        /*
            CharRange range;
        while (tokenIndex_ >= tokenized_.tokens().size()) {
          range.start = nullptr;
          while (range.start == nullptr && pass_ < numPasses_) {
            (*provider_)(range);
            if (range.start == nullptr) {
              ++pass_;
              if (pass_ < numPasses_) {
                provider_->reset();
                tokenizer_.reset();
              }
            }
          }
          if (range.start == nullptr) {
            return false;
          }
          tokenizer_.next(range.start, range.end, tokenized_);
          tokenIndex_ = 0;
        }
        token = tokenized_.tokens()[tokenIndex_++];
        return true;
        */
    }
    
    pub fn new(
        t:          &Tokenizer,
        p:          *mut impl StringProvider,
        num_passes: Option<i32>) -> Self {

        let num_passes: i32 = num_passes.unwrap_or(1);

        todo!();
        /*
            : provider_(p), tokenizer_(t), tokenIndex_(0), numPasses_(numPasses)
        */
    }
}

