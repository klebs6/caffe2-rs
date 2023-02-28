crate::ix!();

#[inline] pub fn store_matrix_in_matrix_market_format<T>(
    m:           i32,
    n:           i32,
    a:           *const T,
    matrix_name: &String)  {

    todo!();
    /*
        using namespace std;
      static set<string> dumped_matrix_names;

      string name(matrix_name);
      string::size_type pos = name.rfind('/');
      if (pos != string::npos) {
        name = name.substr(pos + 1);
      }
      if (dumped_matrix_names.find(name) == dumped_matrix_names.end()) {
        dumped_matrix_names.insert(name);

        FILE* fp = fopen((matrix_name + ".mtx").c_str(), "w");
        if (!fp) {
          return;
        }

        if (is_integral<T>::value) {
          fprintf(fp, "%%%%MatrixMarket matrix array integer general\n");
        } else {
          fprintf(fp, "%%%%MatrixMarket matrix array real general\n");
        }
        fprintf(fp, "%d %d\n", m, n);
        // matrix market array format uses column-major order
        for (int j = 0; j < n; ++j) {
          for (int i = 0; i < m; ++i) {
            if (is_integral<T>::value) {
              fprintf(fp, "%d\n", static_cast<int>(a[j * m + i]));
            } else {
              fprintf(fp, "%f\n", static_cast<float>(a[j * m + i]));
            }
          }
        }

        fclose(fp);
      }
    */
}
