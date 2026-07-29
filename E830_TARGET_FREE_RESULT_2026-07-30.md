# E830 target-free result: curriculum progression passed

The single frozen `E830_curriculum_progression_target_free_v1` source gate ran
from pushed commit `a4b32e44a7c70d6ea9fd825391ae58dfb01ace98` under project `.venv`
Python 3.12.8 and scikit-learn 1.8.0. Runtime was `0.8430311` seconds. It
opened no outcome column or validation environment.

Every preregistered clause passed:

- 636 unique official statements across all eight stages and 22 domains; the
  smallest stage has 54 statements.
- Official leave-one-out expected-stage Spearman `0.7756216`, mean absolute
  stage error `1.1815793`, exact best-stage accuracy `0.2688679`, and
  within-one accuracy `0.7610063`.
- Across all 398 competition objectives, median maximum official similarity
  `0.3020224` and 10th percentile `0.2012611`. Every stage is selected:
  counts from stages 1-8 are `18, 26, 36, 54, 85, 77, 58, 44`.
- Frozen anchors: `17/24` (`0.7083333`) in range, `20/24` (`0.8333333`)
  within one stage, midpoint/expected-stage Spearman `0.8462139`.
- Fixed BGE-base objective-embedding ridge explains only `0.4739721`
  out-of-fold R-squared, with MAE `0.5910205`. The external curriculum
  construction therefore adds substantial structure not linearly recovered by
  v0.5's base objective representation.
- Synthetic proper-score benchmark: control loss `0.6532323`, curriculum
  candidate loss `0.6378295`, gain `0.0154028`.

Bound artifact SHA-256 values:

- objective features:
  `2391a8f5d807323c2802086e8b8f78d126b0115f4ad71531deaac267c453ec19`;
- parsed standards:
  `e14840cb17bcab37b22842187ecd606e8917cfe53ff0a0dc68ebe97ccd27c2ca`;
- anchor audit:
  `35ab2be7cfe9853f270338d726a9b530c50283f1e4e3a5423d47bbcb3e872fde`;
- target-free report:
  `5b53d3f6fd3d34179c4a2f885d6592538d82d77090a74f1682bdb53d93ca9a2b`.

This pass authorizes only a separately frozen outcome screen. It is not
evidence of a competition gain, a production model, or a submission
candidate. `V_joint` and `V_final` remain sealed; no ZIP, upload, or
submission was produced.
