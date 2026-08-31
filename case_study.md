# Case study: Vector — Lead Intelligence Platform

## The problem

Sales teams typically work leads in the order they arrived, not the order they actually matter. Engagement metrics (email opens, clicks) measure interest, but not deal viability — a small company with no real budget can engage heavily and never convert, while a quieter enterprise lead with genuine intent gets deprioritized simply for lack of visible activity. Neither signal alone is enough.

## The approach

Vector combines behavioral and firmographic signal deliberately, rather than scoring on either alone:

1. **Unsupervised tiering (K-Means)** groups leads into hot, warm, cold, or unqualified based on engagement behavior — with an explicit business override: a lead that's highly engaged but too small/low-budget to be a realistic buyer is marked "unqualified" regardless of its cluster, because engagement predicts interest, not deal viability.
2. **Supervised conversion scoring (RandomForestClassifier)** predicts a 0-100 conversion probability from a mix of behavioral features (event counts and velocity, weighted by intent — a demo request outweighs an email open) and firmographic features (company size, ordinally-encoded budget tier, one-hot-encoded source channel).

A business-rule layer turns each score into a specific reason (e.g. "requested a demo," "enterprise budget tier with active engagement") and a recommended action, explicitly overriding a high score when the lead is unqualified or too new to judge reliably.

## Why synthetic data

No client data exists for this demo. The generator (`src/synthetic_data.py`) hides a "true intent" score per lead that drives both engagement-event frequency and conversion probability through a logistic function — the model has to recover a genuinely noisy signal, not read a labeled column.

## An honest result, not a polished one

The conversion classifier's ROC-AUC (0.75) is deliberately reported as lower than Beacon's churn model (0.96) — because the underlying synthetic conversion outcome has more irreducible randomness by design (a probabilistic draw around a combined score, not a hard decay threshold). This mirrors real B2B conversion data, which is noisier than usage-decay data. Reporting a lower, honest number here is more useful to a client than tuning the generator to produce an artificially clean result.

## Results (on the bundled 2,000-lead demo sample)

- 4 lead tiers recovered with a silhouette score of 0.49
- Conversion classifier ROC-AUC of 0.75 on a held-out, stratified test set
- ~14% overall conversion rate, realistic for B2B lead generation
- Every scored lead ships with a specific plain-English reason, not just a number

## What this demonstrates for client work

The same pipeline (`src/pipeline.py`) is built to run unmodified against a real CRM or marketing-automation export — swapping `DATA_MODE=local` and pointing the loader at real leads-and-events tables requires no changes to cleaning, feature engineering, segmentation, scoring, or recommendation logic. This is the third proof point (after Pulse and Beacon) that the same pipeline architecture generalizes across verticals with genuinely different data shapes — single-table event logs, time-series usage, and now two-table relational data.
