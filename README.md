# PriceBook

Quantitative finance library for derivatives pricing and risk analytics. Results validated against standard textbooks and research papers.

PriceBook is the successor to QuantPricer. After the original codebase was lost to a hardware theft, the decision was made to start over from scratch rather than try to reconstruct what was gone. The ideas, the lessons, and the intuition survived — the code didn't. This is the rebuild, done right.

## Getting started

- **`python/notebooks/examples/quickstart.ipynb`** — 20-minute walkthrough (curve → bond → swap → option → serialise → plot)
- **[GUIDE.md](GUIDE.md)** — per-layer API reference with runnable snippets
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — layer structure and dependency rules
- **[RELEASE_NOTES.md](RELEASE_NOTES.md)** — version history

At a glance: 780+ modules, 11,500+ tests, 33 markets, 9 dependency layers, 0 circular deps.

## Structure

```
python/           Python library
```

## License

[MIT](LICENSE) — Bernardo Cohen / deLaPatada Software
