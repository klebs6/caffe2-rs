crate::ix!();

#[test] fn uniform_fill_example_int() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op_1 = core.CreateOperator(
        "UniformIntFill",
        [],
        ["output"],
        min=5,
        max=10,
        shape=(3,3)
    )

    op_2 = core.CreateOperator(
        "UniformIntFill",
        ["shape", "min", "max"],
        ["output"],
        input_as_shape=1
    )

    // Test arg-based op
    workspace.RunOperatorOnce(op_1)
    print("output (op_1):\n", workspace.FetchBlob("output"))

    // Test input-based op
    workspace.ResetWorkspace()
    workspace.FeedBlob("shape", np.array([5,5]))
    workspace.FeedBlob("min", np.array(13, dtype=np.int32))
    workspace.FeedBlob("max", np.array(19, dtype=np.int32))
    workspace.RunOperatorOnce(op_2)
    print("output (op_2):\n", workspace.FetchBlob("output"))

    output (op_1):
     [[ 6 10  7]
     [ 5 10  6]
     [ 7  5 10]]
    output (op_2):
     [[19 13 15 13 13]
     [14 17 14 15 15]
     [17 14 19 13 13]
     [17 18 16 13 18]
     [14 15 16 18 16]]
    */
}

#[test] fn uniform_fill_example_float() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op_1 = core.CreateOperator(
        "UniformFill",
        [],
        ["output"],
        min=5.5,
        max=10.5,
        shape=(3,3)
    )

    op_2 = core.CreateOperator(
        "UniformFill",
        ["shape", "min", "max"],
        ["output"],
        input_as_shape=1
    )

    // Test arg-based op
    workspace.RunOperatorOnce(op_1)
    print("output (op_1):\n", workspace.FetchBlob("output"))

    // Test input-based op
    workspace.ResetWorkspace()
    workspace.FeedBlob("shape", np.array([5,5]))
    workspace.FeedBlob("min", np.array(13.8, dtype=np.float32))
    workspace.FeedBlob("max", np.array(19.3, dtype=np.float32))
    workspace.RunOperatorOnce(op_2)
    print("output (op_2):\n", workspace.FetchBlob("output"))

    output (op_1):
     [[8.894862  8.225005  6.7890406]
     [9.588293  7.1072135 7.7234955]
     [8.210596  6.0202913 9.665462 ]]
    output (op_2):
     [[18.965155 15.603871 15.038921 17.14872  18.134571]
     [18.84237  17.845276 19.214737 16.970337 15.494069]
     [18.754795 16.724329 15.311974 16.962536 18.60965 ]
     [15.186268 15.264773 18.73341  19.077969 14.237255]
     [15.917589 15.844325 16.248466 17.006554 17.502048]]

    */
}
