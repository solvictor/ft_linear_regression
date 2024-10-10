import sys
import json
from tqdm import tqdm
import random
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def estimate_price(θ1, θ0, mileage):
    return θ1 * mileage + θ0


def normalize(value, mean, std_dev):
    return (value - mean) / std_dev


def denormalize(normalized_value, mean, std_dev):
    return normalized_value * std_dev + mean


def train(data, l_rate, epochs):
    θ1 = random.uniform(0, 1)  # Try with 0 and compare
    θ0 = random.uniform(0, 1)
    θ1 = 0
    θ0 = 0
    mileage = data["km"]
    prices = data["price"]
    prices = [normalize(e, prices.mean(), prices.std()) for e in prices]
    mileage = [normalize(e, mileage.mean(), mileage.std()) for e in mileage]
    m = len(data)
    for epoch in tqdm(range(epochs)):
        tmp0 = l_rate * (
            sum(estimate_price(θ1, θ0, mileage[i]) - prices[i] for i in range(m)) / m
        )
        tmp1 = l_rate * (
            sum(
                (estimate_price(θ1, θ0, mileage[i]) - prices[i]) * mileage[i]
                for i in range(m)
            )
            / m
        )
        θ1 -= tmp1
        θ0 -= tmp0
        error = 0  # TODO
    mileage = data["km"]
    prices = data["price"]
    θ1 = θ1 * (prices.std() / mileage.std())
    θ0 = prices.mean() - θ1 * (mileage.mean())
    print(f"{error=}")
    return θ1, θ0


def plot_result(data, θ1, θ0):
    y_line = [(θ1 * x + θ0) for x in data["km"]]
    plt.figure("Linear regression result", figsize=(10, 5))
    plt.scatter(data["km"], data["price"], color="blue", alpha=0.7)
    plt.plot(data["km"], y_line, color="red", label=f"y = {θ1}x + {θ0}")
    plt.xlabel("Kilometers Driven (km)")
    plt.ylabel("Price")
    plt.axhline(0, color="black", lw=0.5, ls="--")
    plt.axvline(0, color="black", lw=0.5, ls="--")
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
        help="Path of the input csv file with training data.",
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

    # TODO training error stop threshold

    args = parser.parse_args()

    try:
        data = pd.read_csv(args.input_file)
        assert "km" in data and "price" in data, "Invalid input data."
        θ1, θ0 = train(data, args.learning_rate, args.epochs)
        if args.plot:
            plot_result(data, θ1, θ0)
        with open(args.output_file, "w") as thetas:
            json.dump({"θ1": θ1, "θ0": θ0}, thetas, indent=4, ensure_ascii=False)
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
