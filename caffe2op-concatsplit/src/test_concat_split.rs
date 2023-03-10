crate::ix!();

#[test] fn split_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Split",
        ["input"],
        ["output_0","output_1","output_2"],
        split=(3,2,4),
        axis=0
    )

    workspace.FeedBlob("input", np.random.randint(10, size=(9)))
    print("input:", workspace.FetchBlob("input"))
    workspace.RunOperatorOnce(op)
    print("output_0:", workspace.FetchBlob("output_0"))
    print("output_1:", workspace.FetchBlob("output_1"))
    print("output_2:", workspace.FetchBlob("output_2"))



    input: [2 2 6 6 6 0 5 7 4]
    output_0: [2 2 6]
    output_1: [6 6]
    output_2: [0 5 7 4]

    */
}

#[test] fn split_by_lengths_op_example1() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "SplitByLengths",
        ["input", "lengths"],
        ["output_0","output_1","output_2"],
        axis=0
    )

    workspace.FeedBlob("input", np.random.randint(10, size=(9)))
    workspace.FeedBlob("lengths", np.array([3,2,4], dtype=np.int32))
    print("input:", workspace.FetchBlob("input"))
    print("lengths:", workspace.FetchBlob("lengths"))
    workspace.RunOperatorOnce(op)
    print("output_0:", workspace.FetchBlob("output_0"))
    print("output_1:", workspace.FetchBlob("output_1"))
    print("output_2:", workspace.FetchBlob("output_2"))

    input: [2 2 6 6 6 0 5 7 4]
    lengths: [3 2 4]
    output_0: [2 2 6]
    output_1: [6 6]
    output_2: [0 5 7 4]

    */
}

#[test] fn split_by_lengths_op_example2() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "SplitByLengths",
        ["input", "lengths"],
        ["output_0","output_1","output_2"],
        axis=0,
        use_scaling_lengths=true,
    )

    workspace.FeedBlob("input", np.random.randint(10, size=(9)))
    workspace.FeedBlob("lengths", np.array([1,1,1], dtype=np.int32))
    print("input:", workspace.FetchBlob("input"))
    print("lengths:", workspace.FetchBlob("lengths"))
    print("output_0:", workspace.FetchBlob("output_0"))
    print("output_1:", workspace.FetchBlob("output_1"))
    print("output_2:", workspace.FetchBlob("output_2"))

    input: [2 2 6 6 6 0 5 7 4]
    lengths: [1 1 1]
    output_0: [2 2 6]
    output_1: [6 6 6]
    output_2: [5 7 4]
    */
}

#[test] fn concat_op_example1() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Concat",
        ["X1",  "X2"],
        ["Y", "split_info"],
        axis=0
    )

    workspace.FeedBlob("X1", np.array([[1,2],[3,4]]))
    workspace.FeedBlob("X2", np.array([[5,6]]))
    print("X1:", workspace.FetchBlob("X1"))
    print("X2:", workspace.FetchBlob("X2"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))
    print("split_info:", workspace.FetchBlob("split_info"))


    X1: [[1 2]
     [3 4]]
    X2: [[5 6]]
    Y: [[1 2]
     [3 4]
     [5 6]]
    split_info: [2 1]
    */
}

#[test] fn concat_op_example2() {

    todo!();
    /*

    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Concat",
        ["X1",  "X2"],
        ["Y", "split_info"],
        add_axis=1,
        axis=3
    )

    workspace.FeedBlob("X1", np.random.randint(10, size=(1, 1, 5, 5))) // NCHW
    workspace.FeedBlob("X2", np.random.randint(10, size=(1, 1, 5, 5))) // NCHW
    print("X1:", workspace.FetchBlob("X1"))
    print("X2:", workspace.FetchBlob("X2"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))
    print("split_info:", workspace.FetchBlob("split_info"))

    X1: [[[[1 8 3 9 0]
       [6 4 6 5 6]
       [3 9 1 9 9]
       [5 1 0 7 7]
       [9 4 0 0 9]]]]
    X2: [[[[7 0 2 6 1]
       [3 9 4 0 3]
       [5 3 8 9 4]
       [3 4 2 1 0]
       [0 8 8 8 1]]]]
    Y: [[[[[1 8 3 9 0]
        [7 0 2 6 1]]

       [[6 4 6 5 6]
        [3 9 4 0 3]]

       [[3 9 1 9 9]
        [5 3 8 9 4]]

       [[5 1 0 7 7]
        [3 4 2 1 0]]

       [[9 4 0 0 9]
        [0 8 8 8 1]]]]]
    split_info: [1 1]

    */
}
