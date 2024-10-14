import sys
import json
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="ft_linear_regression",
        description="Predict a price based on training results.",
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Path of the json file with trained thetas. Defaults to 'thetas.json'.",
        default="thetas.json",
    )

    args = parser.parse_args()

    try:
        θ1 = θ0 = 0.0
        try:
            with open(args.file, "r") as thetas:
                data = json.load(thetas)
                θ1, θ0 = data.get("θ1", 0), data.get("θ0", 0)
        except Exception:
            pass
        mileage = float(input("Enter a mileage: "))
        print(round(θ1 * mileage + θ0, 2))
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
