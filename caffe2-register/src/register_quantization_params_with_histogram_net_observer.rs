crate::ix!();

use crate::{
    NetObserver,
    NetBase
};

/**
  | Set quantization parameters of operators
  | based on min/max collected from OutputMinMaxObserver
  |
  */
pub struct RegisterQuantizationParamsWithHistogramNetObserver {
    base: NetObserver,
}

impl RegisterQuantizationParamsWithHistogramNetObserver {

    pub fn new(
        subject:                  *mut NetBase,
        histogram_file_name:      &String,
        is_weight:                Option<bool>,
        qparams_output_file_name: &String) -> Self {

        let is_weight: bool = is_weight.unwrap_or(false);

        todo!();
        /*
            : NetObserver(subject) 

      ifstream f(histogram_file_name);

      // check the format by looking at the first line
      string first_line, word;
      getline(f, first_line);
      f.seekg(0, f.beg);
      istringstream ist(first_line);
      int nwords_first_line = 0;
      while (ist >> word) {
        ++nwords_first_line;
      }

      ist.str(first_line);
      ist.clear();

      bool new_format = true;
      int op_index, i, nbins;
      string op_type, tensor_name;
      float min, max;
      ist >> op_index >> op_type >> i >> tensor_name >> min >> max >> nbins;
      if (nwords_first_line != nbins + 7) {
        ist.str(first_line);
        ist.clear();
        ist >> op_index >> i >> tensor_name >> min >> max >> nbins;
        if (nwords_first_line == nbins + 6) {
          new_format = false;
        } else {
          LOG(WARNING) << "histogram file " << histogram_file_name
                       << " has an invalid format";
          return;
        }
      }

      // Optionally dump quantization params to file
      ofstream fout;
      if (!qparams_output_file_name.empty()) {
        fout.open(qparams_output_file_name);
        if (!fout) {
          LOG(WARNING) << this << ": can't open " << qparams_output_file_name;
        }
      }

      // parse the input file
      op_index = 0;
      for (auto* op : subject->GetOperators()) {
        for (i = 0; i < op->OutputSize(); ++i) {
          int op_index2, i2;

          if (new_format) {
            f >> op_index2 >> op_type >> i2 >> tensor_name >> min >> max >> nbins;
          } else {
            f >> op_index2 >> i2 >> tensor_name >> min >> max >> nbins;
          }
          LOG_IF(WARNING, op_index2 != op_index)
              << "op index " << op_index2 << " doesn't match with " << op_index;
          LOG_IF(WARNING, tensor_name != op->debug_def().output(i))
              << tensor_name << " in histogram file line " << op_index
              << " doesn't match with operation def " << op->debug_def().output(i);
          LOG_IF(WARNING, i2 != i)
              << "output tensor index " << i2 << " doesn't match with " << i;
          if (new_format) {
            LOG_IF(WARNING, op_type != op->debug_def().type())
                << "operator type " << op_type << " in histogram file line "
                << op_index << " doesn't match with operation def "
                << op->debug_def().type();
          }

          vector<uint64_t> bins;
          for (int j = 0; j < nbins; ++j) {
            uint64_t cnt;
            f >> cnt;
            bins.push_back(cnt);
          }

          Histogram hist = Histogram(min, max, bins);

          LOG(INFO) << "Choosing qparams for " << tensor_name;
          TensorQuantizationParams qparams;
          if (max > min) {
            unique_ptr<QuantizationFactory> qfactory(GetQuantizationFactoryOf(op));
            qparams = qfactory->ChooseQuantizationParams(hist, is_weight);
          } else {
            qparams.scale = 0.1f;
            qparams.precision = 8;
            qparams.zero_point =
                (isinf(min / qparams.scale) || isnan(min / qparams.scale))
                ? 0
                : std::max(
                      0,
                      std::min(
                          int((-min) / qparams.scale),
                          (1 << qparams.precision) - 1));
          }

          if (HasDNNLowPEngine_(*op)) {
            SetStaticQuantizationParams(op, i, qparams);
          }

          if (fout.is_open()) {
            fout << op_index << " " << op_type << " " << i << " " << tensor_name
                 << " " << qparams.Min() << " " << qparams.Max() << " "
                 << qparams.scale << " " << qparams.zero_point << " "
                 << qparams.precision << endl;
          }
        }
        ++op_index;
      }

      if (fout.is_open()) {
        fout.close();
      }
        */
    }
}
