import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_file",
        type=str,
        help="File to write the missing protein sequences into",
    )
    parser.add_argument(
        "datasets",
        choices=["test.csv", "train.csv", "val.csv"],
        nargs="+",
        default=[],
        help="Dataset. Can be one or multiple of test.csv, train.csv, val.csv",
    )
    args = vars(parser.parse_args())

    allSequences = set()
    for datasetName in args["datasets"]:
        print(f"Reading dataset {datasetName}")
        dsPath = f"./data/{datasetName}"
        with open(dsPath, "r") as f:
            sequences = set([line.split(",")[0] for i, line in enumerate(f) if i != 0])
            allSequences = allSequences.union(sequences)

    with open(args["output_file"], "w") as f:
        for seq in allSequences:
            f.write(f"{seq}\n")
