# Stage 2 results: MF + BPR vs LightGCN + BPR

This report covers the six model/negative-proportion configurations and the
most-popular baseline. All numbers are full-catalog ranking metrics on the frozen
temporal split, averaged equally over eligible users. Each configuration uses seeds
42, 43 and 44; tables report mean ± sample standard deviation across seeds.

The LightGCN implementation is described in [lightgcn.md](lightgcn.md). Raw runs are
written under `runs/` (not tracked); the machine-readable aggregation, per-user
statistics and bootstrap results are in `analysis.json`.

## Protocol and training setup

- Shared catalog: 3,492 movies with at least one positive training interaction.
- Validation starts 2001-01-01; test starts 2001-07-01 (UTC).
- 890 validation and 773 test users are eligible under the shared eligibility rules.
- Positive = rating 4–5; explicit negative = rating 1–2; rating 3 is neutral.
- BPR triplets: one negative per positive training interaction per epoch.
- Realized explicit-negative proportions are reported in the tables; fallback users
  lower them slightly relative to the target.
- Both models use embedding dimension 64, learning rate 1e-3, L2 penalty 1e-5,
  batch size 8,192, at most 100 epochs and early stopping with patience 10.
- LightGCN uses two propagation layers. The checkpoint with the best validation
  NDCG@20 is selected before the test evaluation.

## Test metrics

| configuration | realized explicit | best epoch | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity | — | — | 0.0612 | 0.1005 | 0.1758 | 0.1679 |
| MF + BPR, 0% | 0.000 | 38.3 | 0.0673 ± 0.0024 | 0.1149 ± 0.0012 | 0.1962 ± 0.0022 | 0.1902 ± 0.0006 |
| MF + BPR, 25% | 0.2447 ± 0.0001 | 34.7 | 0.0746 ± 0.0004 | 0.1204 ± 0.0016 | 0.1999 ± 0.0027 | 0.1927 ± 0.0009 |
| MF + BPR, 50% | 0.4893 ± 0.0001 | 37.7 | 0.0681 ± 0.0035 | 0.1110 ± 0.0034 | 0.1835 ± 0.0042 | 0.1778 ± 0.0038 |
| LightGCN + BPR, 0% | 0.000 | 68.0 | 0.0691 ± 0.0003 | 0.1186 ± 0.0001 | 0.2014 ± 0.0007 | 0.1949 ± 0.0003 |
| LightGCN + BPR, 25% | 0.2447 ± 0.0001 | 52.7 | 0.0793 ± 0.0021 | 0.1296 ± 0.0003 | 0.2100 ± 0.0030 | 0.2022 ± 0.0013 |
| LightGCN + BPR, 50% | 0.4893 ± 0.0000 | 50.0 | 0.0803 ± 0.0013 | 0.1305 ± 0.0017 | 0.2072 ± 0.0013 | 0.1992 ± 0.0010 |

Validation NDCG@20 at the selected checkpoints:

| configuration | validation NDCG@20 |
| --- | ---: |
| MF + BPR, 0% | 0.1748 ± 0.0008 |
| MF + BPR, 25% | 0.1683 ± 0.0017 |
| MF + BPR, 50% | 0.1524 ± 0.0015 |
| LightGCN + BPR, 0% | 0.1809 ± 0.0004 |
| LightGCN + BPR, 25% | 0.1805 ± 0.0006 |
| LightGCN + BPR, 50% | 0.1739 ± 0.0010 |

LightGCN beats the popularity baseline by a wide margin and beats MF + BPR at every
negative proportion. The margin is small at 0% and larger at 50%. LightGCN also
shows lower seed variance than MF on NDCG@20.

## Confirmatory paired comparisons

Comparisons use test NDCG@20. Each user's metric is averaged across the three seeds
within a configuration, and differences are paired on the 773 users eligible at test.
Confidence intervals are percentile intervals from 10,000 user-level bootstrap
resamples. The Wilcoxon signed-rank test is a complementary check; `Holm p` applies
the prespecified four-test family.

| comparison | mean difference | 95% paired bootstrap CI | Wilcoxon p | Holm p |
| --- | ---: | ---: | ---: | ---: |
| LightGCN − MF, 0% | +0.0048 | [+0.0015, +0.0081] | 0.0084 | 0.0167 |
| LightGCN − MF, 50% | +0.0215 | [+0.0161, +0.0269] | 6.2e-15 | 2.5e-14 |
| MF: 50% − 0% | −0.0124 | [−0.0185, −0.0066] | 0.00015 | 0.00044 |
| LightGCN: 50% − 0% | +0.0043 | [−0.0027, +0.0113] | 0.183 | 0.183 |

Positive differences favour the first configuration. Under this protocol:

- Graph propagation improves NDCG@20 at both 0% and 50% explicit negatives.
- Replacing half of the unrated negatives with explicit dislikes hurts MF.
- For LightGCN, the 50% condition is not significantly different from 0%; the 25%
  condition has the highest point estimate.

## Sensitivity: 25% explicit negatives

The 25% conditions were prespecified as sensitivity analyses. LightGCN at 25% has the
highest NDCG@20 among all configurations (0.2022), while MF at 25% is slightly better
than MF at 0% (0.1927 vs 0.1902). These test-set differences are not part of the
confirmatory family, and the validation ranking for MF is in the opposite direction
(0.1683 at 25% vs 0.1748 at 0%). The discrepancy between validation and test ordering
is a reminder that checkpoint and configuration selection is validation-based while
the test period has a different user/item mix.

## Results by user activity

Groups are based on the number of positive training interactions of the 773 test
users, split at the terciles 61 and 157. Cells are test NDCG@20 averaged across seeds
and users; group sizes are 259 (low), 260 (medium) and 254 (high).

| configuration | low activity | medium activity | high activity |
| --- | ---: | ---: | ---: |
| MF + BPR, 0% | 0.2038 | 0.1896 | 0.1769 |
| MF + BPR, 25% | 0.2056 | 0.1874 | 0.1849 |
| MF + BPR, 50% | 0.1921 | 0.1710 | 0.1701 |
| LightGCN + BPR, 0% | 0.2079 | 0.1950 | 0.1816 |
| LightGCN + BPR, 25% | 0.2176 | 0.2006 | 0.1882 |
| LightGCN + BPR, 50% | 0.2153 | 0.1949 | 0.1873 |

LightGCN improves every activity group. All models score low-activity users higher
than high-activity users on this test period, consistent with high-activity users
having larger, harder target sets.

## Interpretation and limitations

- LightGCN's gain is consistent with neighborhood smoothing: a user's representation
  borrows from co-rated items, which helps when the user-item interaction itself is
  sparse.
- Explicit dislikes are informative at 25%, but pushing to 50% trades away unrated
  coverage. MF degrades at 50%; LightGCN is more robust but still not significantly
  better than its 0% condition.
- Unrated still does not mean disliked. Future positive interactions can be sampled
  as unrated negatives, as allowed by the protocol; the test mask removes them from
  the candidate set but not from training samples.
- The fixed global split leaves 890 validation and 773 test eligible users out of
  6,040. Results describe this population, not all MovieLens users.
- Differences are small in absolute NDCG@20 even when statistically significant
  after Holm correction. Bootstrap intervals condition on the fitted runs; seed
  variability is reported separately in the metric tables.
- Item-popularity subgroup reporting is not included in this stage. The per-user
  artifacts in `runs/*/seed_*/metrics.json` contain everything needed for further
  subgroup or secondary-metric analysis.

## Reproduction

```bash
uv run python -m scripts.run_popularity
for model in mf_bpr lightgcn_bpr; do
  for proportion in 0.0 0.25 0.5; do
    for seed in 42 43 44; do
      uv run python -m scripts.train --model "$model" \
        --explicit-negative-proportion "$proportion" --seed "$seed"
    done
  done
done
uv run python -m scripts.analyze_results
```
