
def print_results(results):
    """Print membership inference attack results in a formatted way."""

    print("Membership Inference Attack Results:")

    for key, value in results.items():
        print(f"{key}: {value:.4f}")