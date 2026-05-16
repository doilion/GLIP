# Medical eval task configs

GLIP task configs (the `--task_config` slot of `tools/test_grounding_net*.py`) for
the TCT_NGC base30 split with PSC-style medical prompts wired in via
`DATASETS.OVERRIDE_CATEGORY` + chunked BERT inference.

## Why a separate config family

GLIP defaults to concatenating **all** class names into a single BERT prompt
(`TEST.CHUNKED_EVALUATION = -1`) and silently truncates anything past
`MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN = 256`. With medical PSC prompts
(diagnostic-system terminology — PSC / Bethesda / Paris / TIS) the concatenated
query for 30 classes hits 400-660 tokens, so **half the classes are silently
dropped** from the language input.

The configs here address that by setting `TEST.CHUNKED_EVALUATION` to a value
small enough that every chunk stays under 256 tokens. The exact value depends
on the prompt verbosity:

| prompt format | total tokens (30 cls) | overflow | recommended chunk | passes/img |
| ------------- | --------------------: | -------: | ----------------: | ---------: |
| `diag_full`   |                  389  |    +133  |  **23**           | 2          |
| `diag_entity` |                  498  |    +242  |  **20**           | 2          |
| `fullnames`   |                  555  |    +299  |  **12**           | 3          |
| `diag_raw`    |                  660  |    +404  |  **11**           | 3          |

Run `tools/check_prompt_token_length.py <override.json>` to re-derive these
numbers for any new prompt set.

## Configs

- **`tct_ngc_base30_diag_raw.yaml`** — richest prompts (e.g. *"PSC Category II:
  Negative — acute neutrophilic inflammation"*). Chunk size 11. Most semantically
  informative; recommended starting point.

To use the other three prompt formats keep the same yaml and override on the CLI:

```bash
# Switch to diag_full prompts (shorter; chunk size 23 fits)
python tools/test_grounding_net.py \
    --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    --weight MODEL/glip_tiny_o365_goldg.pth \
    --task_config configs/medical/tct_ngc_base30_diag_raw.yaml \
    DATASETS.OVERRIDE_CATEGORY "$(jq -c . data/medical/prompts/tct_ngc_base30.diag_full.override.json)" \
    TEST.CHUNKED_EVALUATION 23
```

## Pairing with the medical-prior evaluators

These task configs compose cleanly with the existing medical-prior tools — the
chunked eval handles the language side, the prior tools handle the prediction
side, both run post-hoc on the same `bbox.json` dump:

```bash
# Eval with chunked prompts + exclude-negative AP report
python tools/test_grounding_net_exclude_negative.py \
    --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    --weight MODEL/glip_tiny_o365_goldg.pth \
    --task_config configs/medical/tct_ngc_base30_diag_raw.yaml

# Same plus organ prior + per-organ macro AP
python tools/test_grounding_net_organ_prior.py \
    --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    --weight MODEL/glip_tiny_o365_goldg.pth \
    --task_config configs/medical/tct_ngc_base30_diag_raw.yaml \
    --taxonomy data/medical/tct_ngc_taxonomy.json
```

## Sanity-checking the config

```bash
python tools/check_prompt_token_length.py \
    data/medical/prompts/tct_ngc_base30.diag_raw.override.json \
    --max-query-len 256
```

prints the recommended `TEST.CHUNKED_EVALUATION` value alongside the truncation
diagnosis.
