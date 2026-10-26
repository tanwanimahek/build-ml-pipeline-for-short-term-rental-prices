#!/usr/bin/env python
"""
Download the raw dataset from W&B, apply basic cleaning, and upload the result
as a new W&B artifact. Missing values are intentionally NOT imputed here; that
is handled in the inference pipeline so it also applies at prediction time.
"""
import argparse
import logging

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download the input artifact from W&B
    logger.info("Downloading input artifact %s", args.input_artifact)
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)
    logger.info("Loaded %d rows", df.shape[0])

    # Drop price outliers
    logger.info("Filtering price outliers outside [%s, %s]", args.min_price, args.max_price)
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Converting last_review to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"])

    # Save the cleaned data (index=False avoids an unwanted index column)
    logger.info("Saving cleaned data to clean_sample.csv (%d rows)", df.shape[0])
    df.to_csv("clean_sample.csv", index=False)

    # Upload to W&B
    logger.info("Uploading %s to W&B", args.output_artifact)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
    artifact.wait()

    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name of the input artifact (e.g. sample.csv:latest)",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the cleaned output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum nightly price to keep; rows below are dropped as outliers",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum nightly price to keep; rows above are dropped as outliers",
        required=True,
    )

    args = parser.parse_args()

    go(args)
