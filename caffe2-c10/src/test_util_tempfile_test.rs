crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/test/util/tempfile_test.cpp]

#[cfg(not(_WIN32))]
#[test] fn temp_file_test_matches_expected_pattern() {
    todo!();
    /*
    
      TempFile pattern = make_tempfile("test-pattern-");
      ASSERT_NE(pattern.name.find("test-pattern-"), string::npos);

    */
}

pub fn directory_exists(path: *const u8) -> bool {
    
    todo!();
        /*
            struct stat st;
      return (stat(path, &st) == 0 && (st.st_mode & S_IFDIR));
        */
}

#[test] fn temp_dir_test_try_make_tempdir() {
    todo!();
    /*
    
      optional<TempDir> tempdir = make_tempdir("test-dir-");
      string tempdir_name = tempdir->name;

      // directory should exist while tempdir is alive
      ASSERT_TRUE(directory_exists(tempdir_name.c_str()));

      // directory should not exist after tempdir destroyed
      tempdir.reset();
      ASSERT_FALSE(directory_exists(tempdir_name.c_str()));

    */
}
