import sys
import json
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser


ERROR_FUNCTIONS = {"MAE": abs, "MSE": lambda x: x * x}


def estimate_price(θ1, θ0, mileage):
    return θ1 * mileage + θ0


def normalize(value, mean, std):
    return (value - mean) / std


def train(data, l_rate, epochs, error_fun):
    θ1 = 0
    θ0 = 0
    mileage = data["km"]
    prices = data["price"]
    prices_mean = prices.mean()
    mileage_mean = mileage.mean()
    prices_std = prices.std()
    mileage_std = mileage.std()
    prices = [normalize(p, prices_mean, prices_std) for p in prices]
    mileage = [normalize(m, mileage_mean, mileage_std) for m in mileage]
    m = len(data)
    for _ in tqdm(range(epochs)):
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
    θ1 = θ1 * (prices_std / mileage_std)
    θ0 = prices_mean - θ1 * mileage_mean
    error = (
        sum(
            error_fun(estimate_price(θ1, θ0, km) - p)
            for p, km in zip(data["price"], data["km"])
        )
        / m
    )
    print(f"{error=}")
    return θ1, θ0


def plot_result(data, θ1, θ0):
    y_line = [(θ1 * x + θ0) for x in data["km"]]
    plt.figure("Linear regression results", figsize=(10, 5))
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

    parser.add_argument(
        "--error-function",
        type=str,
        help="Choose the error function. Defaults to abs.",
        choices=list(ERROR_FUNCTIONS),
        default="MAE",
    )

    # TODO training error stop threshold

    args = parser.parse_args()

    try:
        data = pd.read_csv(args.input_file)
        assert "km" in data and "price" in data, "Invalid input data."
        error_fun = ERROR_FUNCTIONS[args.error_function]
        θ1, θ0 = train(data, args.learning_rate, args.epochs, error_fun)
        if args.plot:
            plot_result(data, θ1, θ0)
        with open(args.output_file, "w") as thetas:
            json.dump({"θ1": θ1, "θ0": θ0}, thetas, indent=4, ensure_ascii=False)
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
