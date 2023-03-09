crate::ix!();

#[inline] pub fn format_doc<Def>() -> String {

    todo!();
    /*
        string doc = Def::doc;
      c10::ReplaceAll(doc, "{op}", Def::OpDef::name);
      c10::ReplaceAll(doc, "{op_doc}", Def::OpDef::doc);
      if (strcmp(Def::OpDef::name, "Max") == 0) {
        c10::ReplaceAll(doc, "{extra}", kLengthsMaxExtra);
      } else if (strcmp(Def::OpDef::name, "Mean") == 0) {
        c10::ReplaceAll(doc, "{extra}", kLengthsMeanExtra);
      } else if (strcmp(Def::OpDef::name, "Sum") == 0) {
        c10::ReplaceAll(doc, "{extra}", kLengthsSumExtra);
      } else if (strcmp(Def::OpDef::name, "WeightedSum") == 0) {
        c10::ReplaceAll(doc, "{extra}", kLengthsWeightedSumExtra);
      } else {
        c10::ReplaceAll(doc, "{extra}", " ");
      }
      return doc;
    */
}
