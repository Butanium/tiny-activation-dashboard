"""
Test script for the .serve() functionality.
This creates a minimal dashboard to verify the serving functionality works.
"""

import torch as th
from transformers import AutoTokenizer
from tiny_dashboard import OfflineFeatureCentricDashboard, OnlineFeatureCentricDashboard

def test_offline_serve():
    """Test the offline dashboard serve functionality"""
    print("Testing Offline Dashboard serve()...")

    # Create minimal test data
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Create some fake activation examples
    max_activation_examples = {
        0: [
            (5.2, ["Hello", " world", "!"], [1.0, 5.2, 0.5]),
            (4.8, ["Test", " example"], [4.8, 2.1]),
        ],
        1: [
            (3.5, ["Another", " test"], [3.5, 1.2]),
        ]
    }

    dashboard = OfflineFeatureCentricDashboard(
        max_activation_examples=max_activation_examples,
        tokenizer=tokenizer,
        window_size=50,
        max_examples=10
    )

    # Test serving (non-blocking)
    print("Starting server on http://localhost:8000")
    print("Press Ctrl+C to stop")
    server = dashboard.serve(port=8000, open_browser=False, block=True)


def test_online_serve():
    """Test the online dashboard serve functionality"""
    print("Testing Online Dashboard serve()...")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Create a simple activation function for testing
    def get_feature_activation(text: str, feature_indices: tuple[int, ...]) -> th.Tensor:
        """Dummy activation function that returns random values"""
        tokens = tokenizer.tokenize(text, add_special_tokens=True)
        num_tokens = len(tokens)
        num_features = len(feature_indices)

        # Return random activations
        return th.rand(num_tokens, num_features) * 5.0

    dashboard = OnlineFeatureCentricDashboard(
        get_feature_activation=get_feature_activation,
        tokenizer=tokenizer,
        window_size=50
    )

    # Test serving (non-blocking)
    print("Starting server on http://localhost:8001")
    print("Press Ctrl+C to stop")
    server = dashboard.serve(port=8001, open_browser=False, block=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_serve.py [offline|online]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "offline":
        test_offline_serve()
    elif mode == "online":
        test_online_serve()
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'offline' or 'online'")
        sys.exit(1)
