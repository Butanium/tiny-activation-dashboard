# Tiny Activation Dashboard
A tiny easily hackable implementation of a feature dashboard.
## Installation

```bash
pip install tiny-dashboard
```

## Overview

This repository provides minimal implementations of activations visualization with:
- An online feature dashboard, where you compute and display activations on some custom text
- An offline feature dashboard, which can display precomputed activation examples.

To get an overview of all the features you can check the [demo on colab](https://colab.research.google.com/github/Butanium/tiny-activation-dashboard/blob/main/demo.ipynb)!

Online dashboard demo:
![image](https://github.com/user-attachments/assets/17d176bf-e8e5-471b-bbbf-dc3286f16907)

Offline dashboard demo:
![image](https://github.com/user-attachments/assets/74ab6d98-b10a-4894-a2a3-72f1f20ae7ac)


## Motivation

There are some other good feature activations dashboard tools out there, but I found them very hard to hack on when I wanted to add support for Crosscoders. This implementation is not as complete as https://github.com/jbloomAus/SAEDashboard or even the simplier https://github.com/callummcdougall/sae_vis but in my honest non-biased-at-all opinion, this implementation seems easier to hack on?

If you're looking for a quick and easy to setup tool for feature analysis, this might be the one for you.

## Key Features

Both the offline and online dashboards include:

- Token-level activation highlighting
- Hover tooltips showing token details
- Responsive design
- Save HTML reports

### 1. Offline Feature Exploration

- Analyze pre-computed feature activations
- Visualize max activation examples for specific features
- Expandable text views
- Generate interactive HTML reports

You can either store the max activation examples in a database file, or in a python dictionary.

#### A. Using a python dictionary

```py
from tiny_dashboard.feature_centric_dashboards import OfflineFeatureCentricDashboard

# Create dashboard with pre-computed activations
max_activation_examples: dict[int, list[tuple[float, list[str], list[float]]]] = ...
# max_activation_examples is a dictionary where the keys are feature indices and the values are lists of tuples. Each tuple contains a float (max activation value), a list of strings (the text of the example), and a list of floats (the activation values for each token in the example).

dashboard = OfflineFeatureCentricDashboard(max_activation_examples, tokenizer)
dashboard.display()

# Export to HTML for sharing
feature_to_export = 0
dashboar... [truncated for brevity; full content will be pasted]

## Contributing

Contributions are welcome! Please feel free to improve the minimal design and add some usage examples.

## Citation

If you use Tiny Activation Dashboard in your research or project, please cite it as:

```bibtex
@misc{dumas2024tinyactivationdashboard,
    title = {Tiny Activation Dashboard},
    author = {Clément Dumas},
    year = {2024},
    howpublished = {\url{https://github.com/Butanium/tiny-activation-dashboard}},
}
```