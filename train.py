import sys
import json
import math
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser


ERROR_FUNCTIONS = {"MAE": abs, "MSE": lambda x: x * x}


def estimate_price(θ1, θ0, mileage):
    return θ1 * mileage + θ0


def mean(data):
    return sum(data) / len(data)


def normalize(data):
    data_mean = mean(data)
    data_std = math.sqrt(sum((value - data_mean) ** 2 for value in data) / len(data))
    data = [(value - data_mean) / data_std for value in data]
    return data, data_mean, data_std


def train(data, l_rate, epochs, error_fun, threshold):
    θ0 = 0
    θ1 = 0

    mileages, mileages_mean, mileages_std = normalize(data["km"])
    prices, prices_mean, prices_std = normalize(data["price"])

    m = len(data)
    error = mean(
        [error_fun(estimate_price(θ1, θ0, km) - p) for p, km in zip(data["price"], data["km"])]
    )
    for epoch in tqdm(range(epochs)):
        denormalized_θ1 = θ1 * (prices_std / mileages_std)
        denormalized_θ0 = prices_mean - denormalized_θ1 * mileages_mean
        error = mean(
            [
                error_fun(estimate_price(denormalized_θ1, denormalized_θ0, km) - p)
                for p, km in zip(data["price"], data["km"])
            ]
        )
        if threshold is not None and error <= threshold:
            print("Threshold reached, stoping at epoch", epoch + 1)
            break

        tmp0 = l_rate * (sum(estimate_price(θ1, θ0, mileages[i]) - prices[i] for i in range(m)) / m)
        tmp1 = l_rate * (
            sum((estimate_price(θ1, θ0, mileages[i]) - prices[i]) * mileages[i] for i in range(m))
            / m
        )
        θ0 -= tmp0
        θ1 -= tmp1

    denormalized_θ1 = θ1 * (prices_std / mileages_std)
    denormalized_θ0 = prices_mean - denormalized_θ1 * mileages_mean
    print(f"{error=}")
    return denormalized_θ1, denormalized_θ0


def plot_result(data, θ1, θ0):
    y_line = [(θ1 * x + θ0) for x in data["km"]]
    plt.figure("Linear regression results", figsize=(10, 5))
    plt.scatter(data["km"], data["price"], color="blue", alpha=0.7)
    plt.plot(data["km"], y_line, color="red", label=f"y = {θ1}x + {θ0}")
    plt.xlabel("Kilometers Driven (km)")
    plt.ylabel("Price")
    plt.axhline(0, color="black", lw=0.5, ls="solid")
    plt.axvline(0, color="black", lw=0.5, ls="solid")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="ft_linear_regression",
        description="Train the model on givent data.",
    )

    parser.add_argument(
        "--input-file",
        type=str,
        help="Path of the input csv dataset with training data.",
        default="data/data.csv",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="Path of the output json file with trained thetas.",
        default="thetas.json",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate to use for training.",
        default=0.01,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training iterations.",
        default=250,
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the function and data after training.",
    )

    parser.add_argument(
        "--error-function",
        type=str,
        help="Choose the error function. Defaults to abs.",
        choices=list(ERROR_FUNCTIONS),
        default="MAE",
    )

    parser.add_argument(
        "--error-threshold",
        type=float,
        help="Specify an error threshold below which the training stops.",
    )

    args = parser.parse_args()

    try:
        data = pd.read_csv(args.input_file)
        assert "km" in data and "price" in data, "Invalid input data."

        epochs = args.epochs
        assert epochs >= 0, "Epochs must be a positive integer."

        threshold = args.error_threshold

        error_fun = ERROR_FUNCTIONS[args.error_function]
        θ1, θ0 = train(data, args.learning_rate, epochs, error_fun, threshold)
        if args.plot:
            plot_result(data, θ1, θ0)
        with open(args.output_file, "w") as thetas:
            json.dump(
                {"θ1": θ1, "θ0": θ0},
                thetas,
                indent=4,
                ensure_ascii=False,
            )
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
