crate::ix!();

pub struct Token {
    start_delim_id:  i32,
    start:           *const u8,
    end:             *const u8,
}

pub struct TokenizedString {

    /// holder for strings that have been modified
    modified_strings:  Vec<Arc<String>>,
    tokens:            Vec<Token>,
    last_delim:        i32,
}

impl TokenizedString {
    
    #[inline] pub fn tokens(&self) -> &Vec<Token> {
        
        todo!();
        /*
            return tokens_;
        */
    }
    
    #[inline] pub fn last_delim(&self) -> i32 {
        
        todo!();
        /*
            return lastDelim_;
        */
    }
}

pub struct CharRange {
    start:  *mut u8,
    end:    *mut u8,
}

pub trait StringProvider {
    fn invoke(&mut self, r: &mut CharRange);
    fn reset(&mut self);
}

///-----------------------------
pub struct Tokenizer {

    start_delim_id:  i32,

    /// state of the tokenizer
    leftover:        String,

    /**
      | if we need to skip the first characters of
      | the next batch because
      |
      | e.g. an escape char that was the last
      | character of the last batch.
      */
    to_be_skipped:   i32,

    delim_table:     [i32; 256],
    escape:          u8,
}

impl Tokenizer {
    
    pub fn new(delims: &Vec<u8>, escape: u8) -> Self {
    
        todo!();
        /*
            : escape_(escape) 

      reset();
      std::memset(delimTable_, 0, sizeof(delimTable_));
      for (const auto i : c10::irange(delims.size())) {
        delimTable_[(unsigned char)delims.at(i)] = i + 1;
      }
        */
    }
    
    #[inline] pub fn reset(&mut self)  {
        
        todo!();
        /*
            toBeSkipped_ = 0;
      startDelimId_ = 0;
      leftover_.clear();
        */
    }
    
    #[inline] pub fn next(&mut self, 
        start:     *mut u8,
        end:       *mut u8,
        tokenized: &mut TokenizedString)  {
        
        todo!();
        /*
            tokenized.modifiedStrings_.clear();
      tokenized.tokens_.clear();

      char* currentStart = start;
      std::string* copied = nullptr;
      if (!leftover_.empty()) {
        tokenized.modifiedStrings_.emplace_back(std::make_shared<std::string>());
        copied = tokenized.modifiedStrings_.back().get();
        *copied = std::move(leftover_);
      }

      char* ch;
      for (ch = start + toBeSkipped_; ch < end; ++ch) {
        if (*ch == escape_) {
          if (!copied) {
            tokenized.modifiedStrings_.emplace_back(std::make_shared<std::string>());
            copied = tokenized.modifiedStrings_.back().get();
          }
          copied->append(currentStart, ch);
          currentStart = ch + 1;
          // skip next character, since it's escaped
          ++ch;
          continue;
        }
        int newDelimId = delimTable_[(unsigned char)*ch];
        if (newDelimId > 0) {
          // found delimiter
          tokenized.tokens_.emplace_back();
          auto& token = tokenized.tokens_.back();
          token.startDelimId = startDelimId_;
          if (copied) {
            copied->append(currentStart, ch);
            const char* c_str = copied->data();
            token.start = c_str;
            token.end = c_str + copied->size();
          } else {
            token.start = currentStart;
            token.end = ch;
          }
          currentStart = ch + 1;
          copied = nullptr;
          startDelimId_ = newDelimId - 1;
        }
      }
      tokenized.lastDelim_ = startDelimId_;

      toBeSkipped_ = ch - end;
      if (copied) {
        copied->append(currentStart, end);
        leftover_ = std::move(*copied);
      } else {
        leftover_.assign(currentStart, end);
      }
        */
    }
}
