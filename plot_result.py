import os
from argparse import ArgumentParser
from collections import defaultdict

from matplotlib import pyplot as plt
from matplotlib import colormaps  # type: ignore


def parseArgs():
    parser = ArgumentParser()
    parser.add_argument("input", type=str, help="Input log file")
    parser.add_argument("--prefix", type=str, help="Output prefix", default=None)
    args = parser.parse_args()
    assert os.path.exists(args.input), "Input file does not exist"
    return args


def getErrorRates(logFile: str):
    with open(logFile, "r") as f:
        lines = f.readlines()
    trainErrorRates: dict[int, dict[int, float]] = defaultdict(dict)
    testErrorRates: list[float] = list()
    for line in lines:
        errorRate = float(line.split(" ")[-1])
        if "Train" in line:
            epoch = int(line.split(" ")[0].split("/")[0].strip("["))
            iteration = int(line.split(" ")[1].split("/")[0].strip("["))
            trainErrorRates[epoch][iteration] = errorRate
        elif "Test" in line:
            testErrorRates.append(errorRate)
    return trainErrorRates, testErrorRates


def plotErrorRates(
    trainErrorRates: dict[int, dict[int, float]],
    testErrorRates: list[float],
    outputPrefix: str,
):
    fig = plt.figure(figsize=(8, 5))
    for epoch, errorRates in trainErrorRates.items():
        plt.plot(
            errorRates.keys(),
            errorRates.values(),
            color=colormaps.get_cmap("viridis")(epoch / (len(trainErrorRates) - 1)),
        )
    plt.axhline(
        sum(testErrorRates) / len(testErrorRates),
        color="red",
        linestyle="--",
        label=f"Test Error Rate ${sum(testErrorRates) / len(testErrorRates)*100:.3f}\\%$",
    )
    plt.xlabel("Training steps")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.tight_layout()
    fig.savefig(f"fig/{outputPrefix}.png")
    plt.close(fig)


if __name__ == "__main__":
    args = parseArgs()
    trainErrorRates, testErrorRates = getErrorRates(args.input)
    outputPrefix = args.prefix if args.prefix else os.path.basename(args.input)
    plotErrorRates(trainErrorRates, testErrorRates, outputPrefix)
