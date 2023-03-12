crate::ix!();

#[test] fn elu_functor_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Elu",
        ["X"],
        ["Y"],
        alpha=1.1
    )

    workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32))
    print("X:\n", workspace.FetchBlob("X"), "\n")

    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))

    X:
     [[ 0.35339102  1.1860217  -0.10710736]
     [-3.1173866  -0.1889988  -0.20330353]
     [ 1.8525308  -0.368949    0.506277  ]]

    Y:
     [[ 0.35339102  1.1860217  -0.11172786]
     [-1.0513     -0.18943374 -0.20236646]
     [ 1.8525308  -0.33939326  0.506277  ]]
    */
}
