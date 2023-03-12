crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct TTSparseLengthsSumOp<T,Context,Engine> {
    storage: OperatorStorage,
    context: Context,
    factor_i:  Vec<i32>,
    factor_j:  Vec<i32>,
    ranks:     Vec<i32>,
    l_cumprod: Vec<i32>,
    emb_size:  i32,

    phantom: PhantomData<T>,
    phantomE: PhantomData<Engine>,
}

/**
  | This operator introduce a new, parameter
  | efficient embedding layer, termed
  | TT embedding, which can be plugged in
  | into any model and trained end-to-end.
  | 
  | The benefits of our compressed TT layer
  | are twofold.
  | 
  | Firstly, instead of storing huge embedding
  | matrix, it stores a sequence of much
  | smaller 2-dimensional and 3-dimensional
  | tensors, necessary for reconstructing
  | the required embeddings, which allows
  | compressing the model significantly
  | at the cost of a negligible performance
  | drop.
  | 
  | Secondly, the overall number of parameters
  | can be relatively small (and constant)
  | during the whole training stage, which
  | allows to use larger batches or train
  | efficiently in a case of limited resources.
  |
  */
register_cpu_operator!{TTSparseLengthsSum,         TTSparseLengthsSumOp<f32, CPUContext>}

register_cpu_operator!{TTSparseLengthsSumGradient, TTSparseLengthsSumGradientOp<f32, CPUContext>}

num_inputs!{TTSparseLengthsSum, 5}

num_outputs!{TTSparseLengthsSum, 4}

inputs!{TTSparseLengthsSum, 
    0 => ("core0",          "tensor core 0"),
    1 => ("core1",          "tensor core 1"),
    2 => ("core2",          "tensor core 2"),
    3 => ("index",          "index for embedding"),
    4 => ("lengths",        "segment lengths")
}

outputs!{TTSparseLengthsSum, 
    0 => ("OUTPUT",         "Aggregated tensor"),
    1 => ("core0_output",   "intermediate mm result from core0 for backward path"),
    2 => ("core1_output",   "intermediate mm result from core1 for backward path"),
    3 => ("indices",        "the index for each core")
}

args!{TTSparseLengthsSum, 
    0 => ("factor_i",       "vector<int>: factorization of voc size"),
    1 => ("factor_j",       "vector<int>: factorization of emb size"),
    2 => ("ranks",          "int[] Ranks of cores"),
    3 => ("emb_size",       "int: the size of each embedding entry")
}



impl<T,Context,Engine> TTSparseLengthsSumOp<T,Context,Engine> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            factor_i(this->template GetRepeatedArgument<int>( "factor_i", vector<int>{1, 1, 1})),
            factor_j(this->template GetRepeatedArgument<int>( "factor_j", vector<int>{1, 1, 1})),
            ranks(this->template GetRepeatedArgument<int>( "ranks", vector<int>{1, 1, 1, 1})),
            emb_size(this->template GetSingleArgument<int>("emb_size", 64)) 
        // cumprod of i, used for index slice
        l_cumprod.push_back(1);
        for (size_t i = 1; i < factor_i.size(); ++i) {
          l_cumprod.push_back(l_cumprod[i - 1] * factor_i[i - 1]);
        }
        */
    }
    
    #[inline] pub fn ind_2sub(&mut self, 
        out_factor_index: *mut i64,
        indices:          *const i64,
        len:              i32)  {

        todo!();
        /*
            // TODO: vectorization
        auto N = factor_i.size();
        for (int j = 0; j < len; j++) {
          auto idx = indices[j];
          for (int i = N; i > 0; i--) {
            out_factor_index[j * N + i - 1] = idx / l_cumprod[i - 1];
            idx = idx % l_cumprod[i - 1];
          }
        }
        */
    }
    
    #[inline] pub fn get_slice(&mut self, 
        tgt_slice: &mut Vec<Vec<T>>,
        core:      *const T,
        ind_slice: &Vec<i64>,
        bs:        i32,
        idx:       i32) -> bool {
        
        todo!();
        /*
            // implement the functinality index_select(core, 1, ind_slice)
        auto num_of_elements = ranks[idx] * factor_j[idx] * ranks[idx + 1];
        for (int i = 0; i < bs; i++) {
          memcpy(
              tgt_slice[i].data(),
              core + ind_slice[i] * num_of_elements,
              num_of_elements * sizeof(T));
        }
        return true;
        */
    }

    /**
      | ind: it stores the index to each tensor
      | core bs: the number of indices
      | 
      | GatherAllRows uses two steps to calculate
      | the lengthsum functionality:
      | 
      | 1) it uses tensor train to calculate
      | the embedding for each index.
      | 
      | 2) it sums the embedding for each bag.
      | 
      | In Step 1), it batches all the indices
      | together.
      | 
      | Specifically, for every index, it uses
      | the pre-computed ind of each tensor
      | core to extract the corresponding slice
      | of the core.
      | 
      | Then it does gemm operation sequentially
      | on the slices to produce the embedding
      | result for each index.
      | 
      | In Step 2), it takes the embedding computed
      | in step 1) and apply the sum operation
      | for each bag.
      |
      */
    #[inline] pub fn gather_all_rows(&mut self, 
        ind:      *mut i64,
        bs:       i32,
        x_len:    i32,
        cores:    Vec<*const T>,
        segments: i32,
        lengths:  *const i32,
        out_data: *mut T) -> bool {
        
        todo!();
        /*
            // compute the largest memory consumption of intermediate result
        // TODO: dynamic allocation size: cur_rows*factor_j[i]*ranks[i+1]
        // and also explore the contiguous memory storage for res and int_res
        int max_rank = *max_element(ranks.begin(), ranks.end());
        std::vector<std::vector<T>> res(bs, std::vector<T>(emb_size * max_rank, 0));
        std::vector<std::vector<T>> int_res(
            bs, std::vector<T>(emb_size * max_rank, 0));

        // Store the matrix A
        vector<T*> Y_ptr(bs);
        // Store the intermediate result in each layer
        vector<T*> Z_ptr(bs);

        for (int b = 0; b < bs; b++) {
          Y_ptr[b] = res[b].data();
          Z_ptr[b] = int_res[b].data();
        }

        vector<int64_t> ind_slice(bs);
        int rows = 0;
        for (int i = 0; i < x_len; i++) {
          // slice cur
          for (int j = 0; j < bs; j++) {
            ind_slice[j] = ind[x_len * j + i];
          }
          if (i == 0) {
            GetSlice(res, cores[i], ind_slice, bs, i);
            rows = factor_j[0];
          } else {
            std::vector<std::vector<T>> slice(
                bs, std::vector<T>(ranks[i] * factor_j[i] * ranks[i + 1], 0));
            vector<const T*> X_ptr(bs);
            for (int b = 0; b < bs; b++) {
              X_ptr[b] = slice[b].data();
            }
            GetSlice(slice, cores[i], ind_slice, bs, i);

            math::GemmBatched<T, CPUContext>(
                CblasNoTrans,
                CblasNoTrans,
                bs,
                rows,
                factor_j[i] * ranks[i + 1],
                ranks[i],
                1.0f,
                const_cast<const T**>(Y_ptr.data()),
                X_ptr.data(),
                0.0f,
                Z_ptr.data(),
                &context_);
            for (int b = 0; b < bs; b++) {
              std::memcpy(Y_ptr[b], Z_ptr[b], (emb_size * max_rank) * sizeof(T));
            }
            rows *= factor_j[i];
          }
          // save the intermediate output for backward path
          // shape for the core
          auto shape = vector<int64_t>({bs, rows, ranks[i + 1]});
          if (i < 2) {
            auto* core_data = Output(i + 1, shape, at::dtype<T>());
            T* out_core = core_data->template mutable_data<T>();
            for (int b = 0; b < bs; b++) {
              std::memcpy(
                  out_core + b * rows * ranks[i + 1],
                  Y_ptr[b],
                  rows * ranks[i + 1] * sizeof(T));
            }
          }
        }

        // reduction and store back to output
        vector<int64_t> cum_lengths(segments);
        for (int seg = 0; seg < segments; seg++) {
          cum_lengths[seg] =
              seg == 0 ? lengths[0] : lengths[seg] + cum_lengths[seg - 1];
        }

        int length_idx = 0;
        vector<T> tmp_sum(emb_size, 0.0f);
        for (int i = 0; i <= bs; i++) {
          while ((length_idx < segments) && (i == cum_lengths[length_idx])) {
            // store the tmp_sum into output
            memcpy(
                &out_data[length_idx * emb_size],
                tmp_sum.data(),
                emb_size * sizeof(T));
            length_idx++;
            fill(tmp_sum.begin(), tmp_sum.end(), 0.0f);
          }
          if (i == bs) {
            break;
          }
          transform(
              res[i].begin(),
              res[i].begin() + emb_size,
              tmp_sum.begin(),
              tmp_sum.begin(),
              std::plus<T>());
        }
        return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& dataInput0 = Input(0);
        const auto& dataInput1 = Input(1);
        const auto& dataInput2 = Input(2);
        const auto& indicesInput = Input(3);
        const auto& lengthsInput = Input(4);

        CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
        CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");

        int N = factor_i.size();
        const int64_t M = lengthsInput.size(0);

        auto shape = vector<int64_t>({M, emb_size});
        auto* output = Output(0, shape, at::dtype<T>());
        T* out_data = output->template mutable_data<T>();

        const T* core0 = dataInput0.template data<T>();
        const T* core1 = dataInput1.template data<T>();
        const T* core2 = dataInput2.template data<T>();

        const int* lengths = lengthsInput.template data<int>();

        vector<const T*> cores = {core0, core1, core2};

        const int64_t* indices = indicesInput.template data<int64_t>();

        // Store the factor index for backward path
        auto index_shape = vector<int64_t>({indicesInput.size(), N});
        auto* index_data = Output(3, index_shape, at::dtype<int64_t>());
        int64_t* out_factor_index = index_data->template mutable_data<int64_t>();

        // Store the factorized index for each core
        Ind2Sub(out_factor_index, indices, indicesInput.size());

        return GatherAllRows(
            out_factor_index, indicesInput.size(), N, cores, M, lengths, out_data);
        */
    }
}

