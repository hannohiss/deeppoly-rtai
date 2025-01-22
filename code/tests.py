import logging
import os
import subprocess
import sys

from logger import SimpleLogger

logger = SimpleLogger("Logger", logging.NOTSET)


def check_ground_truth(test_cases_dir: str, spec: str, network: str):
    ground_truth = "gt.txt"
    ground_truth_path = os.path.join(test_cases_dir, ground_truth)
    if not os.path.exists(ground_truth_path):
        logger.error(f"Ground truth file {ground_truth_path} does not exist.")
        return None
    else:
        with open(ground_truth_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.split(",")[:2] == [network, spec]:
                    return line.split(",")[2]
        return "Not found"


def evaluate_networks():
    networks = [
        "fc_linear",
        "fc_base",
        "fc_w",
        "fc_d",
        "fc_dw",
        "fc6_base",
        "fc6_w",
        "fc6_d",
        "fc6_dw",
        "conv_linear",
        "conv_base",
        "conv6_base",
        "conv_d",
        "skip",
        "skip_large",
        "skip6",
        "skip6_large",
    ]
    test_cases_dir = "preliminary_test_cases"  # "test_cases"

    for network in networks:
        logger.debug(f"Evaluating network {network}...")
        network_dir = os.path.join(test_cases_dir, network)
        if not os.path.exists(network_dir):
            logger.debug(
                f"Warning: Directory {network_dir} does not exist, skipping..."
            )
            continue
        specs = os.listdir(network_dir)
        for spec in specs:
            ground_truth = check_ground_truth(test_cases_dir, spec, network)
            spec_path = os.path.join(network_dir, spec)
            logger.debug(f"Evaluating spec {spec} for network {network}... ")
            subprocess.run(
                ["python", "code/verifier.py", "--net", network, "--spec", spec_path]
            )
            print(f"Ground Truth: {ground_truth}")


if __name__ == "__main__":
    evaluate_networks()
