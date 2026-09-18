#!/usr/bin/env python3
"""Quote a customer with a trained model - a terminal preview of the pricing app.

Usage:
    python scripts/quote.py --model glm --DrivAge 22 --VehPower 9
    python scripts/quote.py --model glm --DrivAge 22 --change DrivAge=40 --change BonusMalus=50
    python scripts/quote.py --model glm --DrivAge 22 --explain
    python scripts/quote.py --model xgb --DrivAge 22 --curve DrivAge

Details not given take the model's default (the typical value in the training data).
Run `python scripts/quote.py --model glm --help` to list the details and their ranges.
"""

import argparse
from pathlib import Path

import pandas as pd

from lib.config import DEFAULT_CONFIG, load_config
from lib.pricing_model import load_pricing_model

pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:,.4f}".format)


def parse_value(field, text):
    if field["type"] == "integer":
        return int(text)
    if field["type"] == "number":
        return float(text)
    if text not in field["options"]:
        raise SystemExit(f"{text!r} is not one of {field['options']}")
    return text


def main():
    # first pass: which model (its input fields decide the other options)
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--model", default="glm", help="model name in the model folder (glm or xgb)")
    base_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    known, _ = base_parser.parse_known_args()

    cfg = load_config(known.config)
    model = load_pricing_model(cfg["paths"]["model_dir"] / f"{known.model}.joblib")
    fields = model.input_fields

    parser = argparse.ArgumentParser(
        parents=[base_parser], description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for col, field in fields.items():
        allowed = field["options"] if field["type"] == "category" else f"{field['min']} to {field['max']}"
        parser.add_argument(f"--{col}", help=f"{allowed} (default {field['default']})")
    parser.add_argument("--change", action="append", default=[], metavar="DETAIL=VALUE",
                        help="re-quote with this detail changed (can be repeated)")
    parser.add_argument("--explain", action="store_true",
                        help="show each detail's effect compared with the typical customer")
    parser.add_argument("--curve", metavar="DETAIL", help="show the price across the range of one detail")
    args = parser.parse_args()

    customer = model.default_customer()
    for col, field in fields.items():
        if getattr(args, col) is not None:
            customer[col] = parse_value(field, getattr(args, col))

    quote = model.quote(customer).iloc[0]
    print(f"\nModel: {model.name}")
    print("Customer:", customer)
    print(quote.to_string(), "\n")

    if args.change:
        changed = dict(customer)
        for item in args.change:
            col, value = item.split("=", 1)
            changed[col] = parse_value(fields[col], value)
        new_quote = model.quote(changed).iloc[0]

        comparison = (
            pd.DataFrame({"Before": quote, "After": new_quote})
            .drop("MinimumPremiumApplied")
            .astype(float)
        )
        comparison["Change"] = comparison["After"] - comparison["Before"]
        comparison["Change %"] = 100 * comparison["Change"] / comparison["Before"]
        print("Changed:", ", ".join(args.change))
        print(comparison.to_string(), "\n")

    if args.explain:
        explanation = model.explain(customer)
        print(f"Typical customer's pure premium: {explanation['base_premium']:,.2f}")
        print(explanation["effects"].to_string())
        print(f"This customer's pure premium: {explanation['pure_premium']:,.2f}\n")

    if args.curve:
        curve = model.price_curve(customer, args.curve)
        print(curve[[args.curve, "PurePremium", "AnnualPremium", "MonthlyPremium"]].to_string(index=False))


if __name__ == "__main__":
    main()
