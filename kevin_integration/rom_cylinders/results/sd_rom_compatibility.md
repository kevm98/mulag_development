# SD / ROM Compatibility Note

## SD Checkpoint

- `exists`: `True`
- `path`: `/home/rrlab/rrlab_mulag/kevin_integration/models/sd.pth`
- `input_dim`: `22`
- `output_dim`: `18`
- `hidden_dim`: `128`
- `num_layers`: `2`
- `target_mode`: `delta_normalized`
- `window_size`: `100`
- `has_feature_normalizer`: `True`
- `has_delta_normalizer`: `True`
- `metadata_keys`: `['data_path', 'dropout', 'dynamic_dim', 'epoch', 'feat_mean', 'feat_std', 'fusion_hidden_dim', 'hidden_dim', 'latent_dim', 'optimizer_state_dict', 'output_dim', 'random_seed', 'target_mode', 'teacher_encoder_ckpt', 'val_loss', 'window_size', 'y_delta_mean', 'y_delta_std']`

## Channel Orders

- SD input: `q_1, q_2, q_3, q_4, q_5, qdot_1, qdot_2, qdot_3, qdot_4, qdot_5, dP_1, dP_2, dP_3, dP_4, Fnet_1, Fnet_2, Fnet_3, Fnet_4, ValveCmd_1, ValveCmd_2, ValveCmd_3, ValveCmd_4`
- SD decoded state: `q_next_1, q_next_2, q_next_3, q_next_4, q_next_5, qdot_next_1, qdot_next_2, qdot_next_3, qdot_next_4, qdot_next_5, dP_next_1, dP_next_2, dP_next_3, dP_next_4, Fnet_next_1, Fnet_next_2, Fnet_next_3, Fnet_next_4`
- ROM target: `DeltaExtension_next_1, DeltaExtension_next_2, DeltaExtension_next_3, DeltaExtension_next_4, DeltaDP_next_1, DeltaDP_next_2, DeltaDP_next_3, DeltaDP_next_4`

## Compatibility Summary

- SD input order is [q(5), qdot(5), dP(4), Fnet(4), valve_cmd(4)].
- SD decoded output order is [q_next(5), qdot_next(5), dP_next(4), Fnet_next(4)].
- ROM target order is [DeltaExtension_next(4), DeltaDP_next(4)].
- To compare SD with ROM pressure targets, decode SD state first and compare dP_next - current_dP to ROM DeltaDP_next.
- SD does not predict cylinder DeltaExtension; it cannot replace the learned cylinder ROM motion target.
- The practical first pipeline should use a ROM extension-only target while pressure prediction remains unvalidated.
- Do not use SD as ROM feedback or input until units, normalization, timestep, sign convention, and target_mode are validated on synchronized held-out samples.

## Practical Pipeline Decision

- `Option A`: ROM predicts DeltaExtension only; SD/observer provides dP state.
- `Option B`: ROM predicts DeltaExtension and DeltaDP; SD remains a passive consistency checker.
- `Option C`: ROM predicts DeltaExtension and DeltaDP; SD provides residual/blended correction.
- Recommended first option: `Option A`

## Later Roles

- passive observer
- dP/Fnet comparison channel
- optional Force/Fnet feature provider for a Force-augmented ROM
- small blended correction only after offline compatibility validation

Decision: **Use Option A as the first offline selection target; do not merge SD into ROM training or Isaac ROM integration yet.**
