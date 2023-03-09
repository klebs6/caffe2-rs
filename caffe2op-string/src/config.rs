crate::ix!();

/**
  | Computes the element-wise string prefix
  | of the string tensor.
  | 
  | Input strings that are shorter than
  | prefix length will be returned unchanged.
  | 
  | -----------
  | @note
  | 
  | Prefix is computed on number of bytes,
  | which may lead to wrong behavior and
  | potentially invalid strings for variable-length
  | encodings such as utf-8.
  |
  */
register_cpu_operator!{StringPrefix, StringElementwiseOp<Prefix>}

should_not_do_gradient!{StringPrefix}

num_inputs!{StringPrefix, 1}

num_outputs!{StringPrefix, 1}

inputs!{StringPrefix, 
    0 => ("strings", "Tensor of std::string.")
}

outputs!{StringPrefix, 
    0 => ("prefixes", "Tensor of std::string containing prefixes for each input.")
}

args!{StringPrefix, 
    0 => ("length", "Maximum size of the prefix, in bytes.")
}

/**
  | Computes the element-wise string suffix
  | of the string tensor.
  | 
  | Input strings that are shorter than
  | suffix length will be returned unchanged.
  | 
  | -----------
  | @note
  | 
  | Prefix is computed on number of bytes,
  | which may lead to wrong behavior and
  | potentially invalid strings for variable-length
  | encodings such as utf-8.
  |
  */
register_cpu_operator!{StringSuffix, StringElementwiseOp<Suffix>}

should_not_do_gradient!{StringSuffix}

num_inputs!{StringSuffix, 1}

num_outputs!{StringSuffix, 1}

inputs!{StringSuffix, 
    0 => ("strings", "Tensor of std::string.")
}

outputs!{StringSuffix, 
    0 => ("suffixes", "Tensor of std::string containing suffixes for each output.")
}

args!{StringSuffix, 
    0 => ("length", "Maximum size of the suffix, in bytes.")
}

/**
  | Performs the starts-with check on each
  | string in the input tensor.
  | 
  | Returns tensor of boolean of the same
  | dimension of input.
  |
  */
register_cpu_operator!{StringStartsWith,
    StringElementwiseOp<StartsWith, FixedType<bool>>}

should_not_do_gradient!{StringStartsWith}

num_inputs!{StringStartsWith, 1}

num_outputs!{StringStartsWith, 1}

inputs!{StringStartsWith, 
    0 => ("strings", "Tensor of std::string.")
}

outputs!{StringStartsWith, 
    0 => ("bools", "Tensor of bools of same shape as input.")
}

args!{StringStartsWith, 
    0 => ("prefix", "The prefix to check input strings against.")
}

/**
  | Performs the ends-with check on each
  | string in the input tensor.
  | 
  | Returns tensor of boolean of the same
  | dimension of input.
  |
  */
register_cpu_operator!{StringEndsWith,
    StringElementwiseOp<EndsWith, FixedType<bool>>}

num_inputs!{StringEndsWith, 1}

num_outputs!{StringEndsWith, 1}

inputs!{StringEndsWith, 
    0 => ("strings", "Tensor of std::string.")
}

outputs!{StringEndsWith, 
    0 => ("bools", "Tensor of bools of same shape as input.")
}

args!{StringEndsWith, 
    0 => ("suffix", "The suffix to check input strings against.")
}

should_not_do_gradient!{StringEndsWith}

/**
  | Performs equality check on each string
  | in the input tensor.
  | 
  | Returns tensor of booleans of the same
  | dimension as input.
  |
  */
register_cpu_operator!{StringEquals,
    StringElementwiseOp<StrEquals, FixedType<bool>>}

num_inputs!{StringEquals, 1}

num_outputs!{StringEquals, 1}

inputs!{StringEquals, 
    0 => ("strings", "Tensor of std::string.")
}

outputs!{StringEquals, 
    0 => ("bools", "Tensor of bools of same shape as input.")
}

args!{StringEquals, 
    0 => ("text", "The text to check input strings equality against.")
}

should_not_do_gradient!{StringEquals}

/**
  | Takes a 1-D or a 2-D tensor as input and
  | joins elements in each row with the provided
  | delimiter.
  | 
  | Output is a 1-D tensor of size equal to
  | the first dimension of the input.
  | 
  | Each element in the output tensor is
  | a string of concatenated elements corresponding
  | to each row in the input tensor.
  | 
  | For 1-D input, each element is treated
  | as a row.
  |
  */
register_cpu_operator!{StringJoin, StringJoinOp<CPUContext>}

num_inputs!{StringJoin, 1}

num_outputs!{StringJoin, 1}

inputs!{StringJoin, 
    0 => ("input", "1-D or 2-D tensor")
}

outputs!{StringJoin, 
    0 => ("strings", "1-D tensor of strings created by joining row elements from the input tensor.")
}

args!{StringJoin, 
    0 => ("delimiter", "Delimiter for join (Default: ,)."),
    1 => ("axis", "Axis for the join (either 0 or 1)")
}

should_not_do_gradient!{StringJoin}
