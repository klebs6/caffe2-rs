crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/include/pack_block_sparse.h]

pub struct BCSRMatrix {

    #[cfg(not(_WIN32))] col_indices: Vec<u32,AlignedAllocator<u32,16>>,
    #[cfg(not(_WIN32))] row_values:  Vec<u32,AlignedAllocator<u32,16>>,
    #[cfg(not(_WIN32))] values:      Vec<u8,AlignedAllocator<u8,16>>,

    #[cfg(_WIN32)]      col_indices: Vec<u32>,
    #[cfg(_WIN32)]      row_values:  Vec<u32>,
    #[cfg(_WIN32)]      values:      Vec<u8>,

    /**
      | input features block size
      |
      */
    col_block_size: u32,


    /**
      | output features block size
      |
      */
    row_block_size: u32,
}

impl BCSRMatrix {
    
    pub fn print(&mut self)  {
        
        todo!();
        /*
            cout << "row block size:" << row_block_size << endl;
        cout << "col block size:" << col_block_size << endl;
        cout << "row ptr\n";
        for (const auto& t : row_values) {
          cout << t << ", ";
        }
        cout << endl;
        cout << "col indices\n";
        for (const auto& t : col_indices) {
          cout << t << ", ";
        }
        cout << endl;
        cout << "Actual values\n";
        for (const auto& t : values) {
          cout << (u32)t << ", ";
        }
        cout << endl;
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/pack_block_sparse.cc]

pub fn generate_block_csr_matrix(
    a:              *const u8,
    N:              Size,
    K:              Size,
    row_block_size: u32,
    col_block_size: u32,
    zero_points:    *const u8) -> Box<BCSRMatrix> {

    todo!();
        /*
            assert(K > 0);
      unique_ptr<BCSRMatrix> bcsr_mat_ptr = make_unique<BCSRMatrix>();
      auto& bcsr_mat = *bcsr_mat_ptr;
      const u32 num_row_blocks = (N + row_block_size - 1) / row_block_size;
      // K must be > 0
      const u32 num_col_blocks = (K + col_block_size - 1) / col_block_size;

      bcsr_mat.row_values.reserve(num_row_blocks);
      u32 num_nnz_blocks{0};
      bcsr_mat.row_values.push_back(num_nnz_blocks);
      for (u32 i = 0; i < num_row_blocks; ++i) {
        for (u32 j = 0; j < num_col_blocks; ++j) {
          bool block_zero{true};
          for (u32 ib = 0; ib < row_block_size; ++ib) {
            u32 row_index = i * row_block_size + ib;
            if PYTORCH_QNNP_UNLIKELY(row_index >= N) {
              break;
            }
            for (u32 jb = 0; jb < col_block_size; ++jb) {
              u32 col_index = j * col_block_size + jb;
              if PYTORCH_QNNP_UNLIKELY(col_index >= K) {
                goto block_scanned;
              }
              if (*(a + row_index * K + col_index) != zero_points[row_index]) {
                block_zero = false;
                goto block_scanned;
              }
            }
          }
    block_scanned:
          if (!block_zero) {
            bcsr_mat.col_indices.push_back(j);
            num_nnz_blocks++;
            for (u32 ib = 0; ib < row_block_size; ++ib) {
              u32 row_index = i * row_block_size + ib;
              if PYTORCH_QNNP_UNLIKELY(row_index >= N) {
                for (; row_index < (num_row_blocks * row_block_size); row_index++) {
                  for (u32 jb = 0; jb < col_block_size; ++jb) {
                    bcsr_mat.values.push_back(zero_points[N-1]);
                  }
                }
                break;
              }
              for (u32 jb = 0; jb < col_block_size; ++jb) {
                u32 col_index = j * col_block_size + jb;
                if PYTORCH_QNNP_UNLIKELY(col_index >= K) {
                  bcsr_mat.values.push_back(zero_points[row_index]);
                } else {
                  u8 val = *(a + row_index * K + col_index);
                  bcsr_mat.values.push_back(val);
                }
              }
            }
          }
        }
        bcsr_mat.row_values.push_back(num_nnz_blocks);
      }
      bcsr_mat.row_block_size = row_block_size;
      bcsr_mat.col_block_size = col_block_size;
      return bcsr_mat_ptr;
        */
}
