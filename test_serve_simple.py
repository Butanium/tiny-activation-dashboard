"""
Simple test script for the .serve() functionality without requiring model downloads.
"""

import torch as th
from tiny_dashboard import OfflineFeatureCentricDashboard, OnlineFeatureCentricDashboard


class MockTokenizer:
    """Mock tokenizer for testing"""
    bos_token = "<s>"

    def tokenize(self, text, add_special_tokens=False):
        # Simple whitespace tokenization
        tokens = text.split()
        if add_special_tokens:
            tokens = [self.bos_token] + tokens
        return tokens

    def convert_ids_to_tokens(self, ids):
        return [f"token_{i}" for i in ids]

    def convert_tokens_to_ids(self, tokens):
        # Simple hash-based ID generation
        return [hash(token) % 50000 for token in tokens]


def test_offline_serve():
    """Test the offline dashboard serve functionality"""
    print("Testing Offline Dashboard serve()...")

    tokenizer = MockTokenizer()

    # Create some fake activation examples
    max_activation_examples = {
        0: [
            (5.2, ["Hello", " world", "!"], [1.0, 5.2, 0.5]),
            (4.8, ["Test", " example"], [4.8, 2.1]),
        ],
        1: [
            (3.5, ["Another", " test"], [3.5, 1.2]),
        ],
        42: [
            (6.1, ["Feature", " forty", " two"], [2.0, 6.1, 3.0]),
        ]
    }

    dashboard = OfflineFeatureCentricDashboard(
        max_activation_examples=max_activation_examples,
        tokenizer=tokenizer,
        window_size=50,
        max_examples=10
    )

    # Test serving (blocking)
    print("Starting server on http://localhost:8002")
    print("Press Ctrl+C to stop")
    server = dashboard.serve(port=8002, open_browser=False, block=True)


def test_online_serve():
    """Test the online dashboard serve functionality"""
    print("Testing Online Dashboard serve()...")

    tokenizer = MockTokenizer()

    # Create a simple activation function for testing
    def get_feature_activation(text: str, feature_indices: tuple[int, ...]) -> th.Tensor:
        """Dummy activation function that returns random values"""
        tokens = tokenizer.tokenize(text, add_special_tokens=True)
        num_tokens = len(tokens)
        num_features = len(feature_indices)

        # Return random activations scaled by feature index
        activations = th.rand(num_tokens, num_features) * 5.0
        # Make different features have different patterns
        for i, feat_idx in enumerate(feature_indices):
            activations[:, i] *= (1 + feat_idx * 0.1)
        return activations

    dashboard = OnlineFeatureCentricDashboard(
        get_feature_activation=get_feature_activation,
        tokenizer=tokenizer,
        window_size=50
    )

    # Test serving (blocking)
    print("Starting server on http://localhost:8001")
    print("Press Ctrl+C to stop")
    server = dashboard.serve(port=8001, open_browser=False, block=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_serve_simple.py [offline|online]")
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
