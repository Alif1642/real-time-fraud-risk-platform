# Data Dictionary

This project intentionally consumes a focused subset of IEEE-CIS columns to reduce memory pressure. The raw files remain external to Git.

| Field | Role |
|---|---|
| TransactionID | Join key / response identifier; excluded from model features |
| isFraud | Binary target; training only |
| TransactionDT | Relative transaction time used for chronological validation and time features |
| TransactionAmt | Transaction amount |
| ProductCD | Product code |
| card1-card6 | Card-related anonymized attributes |
| addr1-addr2 | Anonymized address attributes |
| dist1 | Distance-like anonymized feature |
| P_emaildomain / R_emaildomain | Purchaser / recipient email-domain features |
| C1, C2, D1, D2 | Anonymized IEEE-CIS count/time features |
| DeviceType / DeviceInfo | Identity-side device attributes |
| id_01 / id_02 / id_31 | Identity-side anonymized/browser attributes |

## Engineered features
- `TransactionAmt_log1p`: log-transformed amount.
- `TransactionAmt_deviation`: amount minus train-window median.
- `TransactionAmt_percentile_band`: train-window amount band.
- `*_frequency`: train-window category frequency mappings.
- `hour`, `day`, `week`, `day_of_week`: derived from `TransactionDT`.
- `transaction_count_by_card`: train-window card occurrence count.
- `average_amount_by_card`: train-window mean amount per card.
- `amount_deviation_from_card_average`: current amount minus learned card mean.
- `transaction_frequency`: proxy based on learned card frequency.
- `device_change_indicator`, `email_change_indicator`: neutral placeholders for online stateful features; production systems should supply them from event history.
