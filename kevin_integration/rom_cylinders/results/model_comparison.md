# ROM Model Comparison

| Name | Type | Target Mode | Force | Ext RMSE | Ext Sign Acc | dP RMSE | Teacher/Rollout 0.5s Ext RMSE | 1s Ext RMSE | Unstable Fraction |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| zero | zero | extension_only | False | 0.015586579218506813 | 0.12361791614273945 | None | 0.0966777910617598 | 0.08720622752811481 | 0.0 |
| persistence | persistence | extension_only | False | 0.007652221713215113 | 0.8998147930203604 | None | 0.12248013365879484 | 0.13796965305100092 | 0.0 |
| gru_core_seed42 | gru | extension_only | False | 0.015528205782175064 | 0.6626321690276943 | None | 0.08109943173345746 | 0.10461300567784745 | 0.0 |

Selection gates: beat zero and persistence on locked-test DeltaExtension RMSE, beat persistence on teacher-pressure rollout at 0.5 s and 1.0 s, preserve sign behavior for every cylinder, avoid NaN/Inf and state-limit violations, pass inference parity, avoid unavailable Force dependencies, and remain stable across runs. Do not select an extension+pressure model over extension-only unless pressure rollout is demonstrably stable and useful.