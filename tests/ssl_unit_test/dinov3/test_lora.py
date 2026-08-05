# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 LoRA unit tests — the Stage-1 gate ladder (G1.1 – G1.7).

Each test maps to one invariant from ``docs/plans/dinov3_lora_devbox_plan.md``:

======  ==============================================================================
G1.1    Identity: injected LoRA (B=0) forward == stock forward
G1.2    Merge parity: forward(merge(m)) ~= forward(m) after random A/B
G1.3    EMA alignment: student/teacher ``named_parameters()`` name lists identical
G1.4    Freeze audit: ``requires_grad`` true only for ``lora_*``, heads, ``mask_token``
G1.5    Key stability: backbone keys == stock keys + ``lora_*``; timm round-trip unaffected
G1.6    Gram-teacher sync with a LoRA-injected teacher succeeds
G1.7    Optimizer groups: LoRA params land in the right ``blocks.N`` layer-id groups
======  ==============================================================================

The backbone tests build a *tiny* ViT (2 blocks, 64-dim) on CPU in fp32 and disable
``use_custom_attention``: the xformers ``memory_efficient_attention`` path hard-casts q/k/v to
``.half()``, which would swamp the 1e-6 identity tolerance these gates are about.
"""

import pytest
from omegaconf import OmegaConf
import torch
from torch import nn
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy

from nvidia_tao_pytorch.config.dinov3.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.dinov3.model.lora import (
    LoRALinear,
    inject_lora,
    merge_lora,
    has_lora,
    is_lora_key,
    strip_lora_keys,
    lora_parameter_report,
)
from nvidia_tao_pytorch.ssl.dinov3.model.pl_model import DinoV3PlModel
from nvidia_tao_pytorch.ssl.dinov3.model.vit import DinoV3VisionTransformer
from nvidia_tao_pytorch.ssl.dinov3.utils.checkpoint_remap import timm_to_tao, tao_to_timm

# Tiny ViT so the whole gate ladder runs on CPU in seconds.
TINY = dict(
    img_size=32, patch_size=16, embed_dim=64, depth=2, num_heads=4,
    init_values=1e-5, drop_path_schedule="uniform", num_classes=0,
    drop_path_rate=0.0, register_tokens=2, use_custom_attention=False,
)
BATCH_SIZE = 2
RANK = 4
ALPHA = 8.0

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA GPU")


def _tiny_backbone(seed=0):
    """Build a small fp32 DINOv3 ViT on CPU for the structural gates."""
    torch.manual_seed(seed)
    return DinoV3VisionTransformer(**TINY).eval()


def _tiny_input():
    """Deterministic image batch matching the tiny ViT's input size."""
    torch.manual_seed(123)
    return torch.randn(BATCH_SIZE, 3, TINY["img_size"], TINY["img_size"])


def _randomize_lora_(module):
    """Give every adapter a non-trivial B (init is zero, i.e. identity)."""
    generator = torch.Generator().manual_seed(7)
    for submodule in module.modules():
        if isinstance(submodule, LoRALinear):
            with torch.no_grad():
                submodule.lora_A.copy_(torch.randn(submodule.lora_A.shape, generator=generator) * 0.05)
                submodule.lora_B.copy_(torch.randn(submodule.lora_B.shape, generator=generator) * 0.05)


# --------------------------------------------------------------------------------------
# G1.1 — identity at init
# --------------------------------------------------------------------------------------

@pytest.mark.ssl_unit
def test_lora_linear_is_identity_at_init():
    """A freshly wrapped LoRALinear reproduces its base nn.Linear bit-for-bit (B=0)."""
    torch.manual_seed(0)
    base = nn.Linear(32, 16)
    x = torch.randn(4, 32)
    expected = base(x)

    lora = LoRALinear.from_linear(base, rank=RANK, alpha=ALPHA, dropout=0.0)

    # The adopted Parameter objects are the *same* tensors, not copies.
    assert lora.weight is base.weight
    assert lora.bias is base.bias
    assert torch.equal(lora.lora_B, torch.zeros_like(lora.lora_B))
    torch.testing.assert_close(lora(x), expected, rtol=0, atol=0)


@pytest.mark.ssl_unit
def test_injection_identity():
    """G1.1: injecting LoRA into a backbone leaves its forward unchanged (<=1e-6)."""
    backbone = _tiny_backbone()
    x = _tiny_input()
    with torch.no_grad():
        before = backbone(x)

    injected = inject_lora(backbone, rank=RANK, alpha=ALPHA, dropout=0.0)
    assert injected, "expected at least one LoRA target to be injected"
    assert has_lora(backbone)

    with torch.no_grad():
        after = backbone(x)

    for key in ("x_norm_clstoken", "x_norm_patchtokens"):
        torch.testing.assert_close(after[key], before[key], rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------------------
# G1.2 — merge parity
# --------------------------------------------------------------------------------------

@pytest.mark.ssl_unit
def test_merge_parity():
    """G1.2: after random A/B, merging folds the delta in without changing the forward."""
    backbone = _tiny_backbone()
    inject_lora(backbone, rank=RANK, alpha=ALPHA, dropout=0.0)
    _randomize_lora_(backbone)

    x = _tiny_input()
    with torch.no_grad():
        unmerged = backbone(x)

    # Sanity: the randomized adapter actually changed the function, so parity is meaningful.
    stock = _tiny_backbone()
    with torch.no_grad():
        stock_out = stock(x)
    assert not torch.allclose(unmerged["x_norm_clstoken"], stock_out["x_norm_clstoken"], atol=1e-5), \
        "randomized LoRA did not perturb the backbone; merge parity would be vacuous"

    merged_count = merge_lora(backbone)
    assert merged_count == len(list(m for m in backbone.modules() if isinstance(m, LoRALinear)))

    with torch.no_grad():
        merged = backbone(x)

    for key in ("x_norm_clstoken", "x_norm_patchtokens"):
        torch.testing.assert_close(merged[key], unmerged[key], rtol=1e-5, atol=1e-5)


@pytest.mark.ssl_unit
def test_merge_is_idempotent_and_matches_delta_weight():
    """Merging twice is a no-op, and the folded delta equals (alpha/r) * B @ A."""
    torch.manual_seed(0)
    base = nn.Linear(16, 8, bias=False)
    lora = LoRALinear.from_linear(base, rank=RANK, alpha=ALPHA)
    original_weight = lora.weight.detach().clone()
    _randomize_lora_(lora)

    expected_delta = lora.delta_weight().detach().clone()
    lora.merge()
    torch.testing.assert_close(lora.weight.data - original_weight, expected_delta, rtol=1e-6, atol=1e-6)

    weight_after_first = lora.weight.detach().clone()
    lora.merge()  # idempotent
    torch.testing.assert_close(lora.weight.data, weight_after_first, rtol=0, atol=0)


# --------------------------------------------------------------------------------------
# G1.5 — key stability
# --------------------------------------------------------------------------------------

@pytest.mark.ssl_unit
def test_state_dict_keys():
    """G1.5: injection adds only lora_* keys; every stock key survives unchanged."""
    stock_keys = set(_tiny_backbone().state_dict())

    backbone = _tiny_backbone()
    inject_lora(backbone, rank=RANK, alpha=ALPHA)
    injected_keys = set(backbone.state_dict())

    added = injected_keys - stock_keys
    assert not (stock_keys - injected_keys), "injection dropped stock backbone keys"
    assert added, "injection added no keys"
    assert all(is_lora_key(k) for k in added), f"non-LoRA keys appeared: {sorted(added - set(filter(is_lora_key, added)))}"
    assert strip_lora_keys(backbone.state_dict()).keys() == stock_keys

    # Three lora entries per injected module: lora_A, lora_B and the persisted scale.
    n_lora_modules = sum(1 for m in backbone.modules() if isinstance(m, LoRALinear))
    assert len(added) == 3 * n_lora_modules


@pytest.mark.ssl_unit
def test_convert_merges_adapters_into_exported_weights(tmp_path):
    """End-to-end: `dinov3 convert` must export the *adapted* weights, not the frozen base.

    Exercises the real converter rather than stripping adapter keys in the test. With nonzero
    A/B the exported weight must equal ``W + (alpha/r) * B @ A`` and no ``lora_*`` key may
    survive -- previously the adapters either failed timm validation or were written straight
    into a supposedly timm-format file, and export silently shipped the unadapted model.
    """
    from nvidia_tao_pytorch.ssl.dinov3.utils.checkpoint_remap import convert_ssl_to_timm

    backbone = _tiny_backbone()
    inject_lora(backbone, rank=RANK, alpha=ALPHA)
    _randomize_lora_(backbone)

    # Expected merged weights, computed independently of the converter.
    expected = {}
    for name, module in backbone.named_modules():
        if isinstance(module, LoRALinear):
            expected[f"{name}.weight"] = (
                module.weight.detach() + module.delta_weight().detach()
            ).clone()
    assert expected, "no adapters to merge; the test would be vacuous"

    src = tmp_path / "ssl_ckpt.pth"
    dst = tmp_path / "backbone.safetensors"
    torch.save({"state_dict": {f"teacher.backbone.{k}": v
                               for k, v in backbone.state_dict().items()}}, src)

    converted = convert_ssl_to_timm(
        str(src), str(dst), source="teacher", validate=False,
        timm_model_name="vit_small_patch16_dinov3",
    )

    assert not [k for k in converted if is_lora_key(k)], "lora_* keys leaked into the export"
    for key, want in expected.items():
        assert key in converted, f"{key} missing from the converted backbone"
        torch.testing.assert_close(converted[key], want, rtol=1e-6, atol=1e-6)

    # And the merge must have actually changed something relative to the frozen base.
    stock = _tiny_backbone().state_dict()
    changed = [k for k in expected if not torch.allclose(converted[k], stock[k], atol=1e-6)]
    assert changed, "converted weights equal the frozen base; the adapters were not applied"


@pytest.mark.ssl_unit
def test_timm_remap_roundtrip_unaffected_by_lora():
    """G1.5: the timm<->TAO key translation is untouched for base keys, and skips lora keys."""
    backbone = _tiny_backbone()
    inject_lora(backbone, rank=RANK, alpha=ALPHA)

    for key in strip_lora_keys(backbone.state_dict()):
        assert timm_to_tao(tao_to_timm(key)) == key, f"round-trip broke for {key}"

    # LoRA keys are recognized as adapter keys, so convert can merge/drop them before
    # translation. `lora_scaling` is the persisted alpha/rank buffer and counts as one.
    lora_keys = [k for k in backbone.state_dict() if is_lora_key(k)]
    assert lora_keys and all(
        k.endswith(("lora_A", "lora_B", "lora_scaling")) for k in lora_keys
    ), sorted(lora_keys)


@pytest.mark.ssl_unit
def test_num_last_blocks_and_target_modules_are_respected():
    """Only the requested projections in the requested blocks get adapters."""
    backbone = _tiny_backbone()
    injected = inject_lora(backbone, rank=RANK, alpha=ALPHA,
                           target_modules=["qkv"], num_last_blocks=1)

    assert injected == ["blocks.1.attn.qkv"]
    assert isinstance(backbone.blocks[1].attn.qkv, LoRALinear)
    assert not isinstance(backbone.blocks[1].attn.proj, LoRALinear)
    assert not isinstance(backbone.blocks[0].attn.qkv, LoRALinear)


@pytest.mark.ssl_unit
def test_injection_is_idempotent():
    """Re-injecting must not disarm adapters added by an earlier call.

    Freezing the whole backbone on every call switched off existing ``lora_A``/``lora_B``,
    and since already-injected modules are skipped nothing turned them back on -- so a second
    call, or a staged ``qkv`` then ``proj`` injection, silently trained only the heads.
    """
    backbone = _tiny_backbone()
    inject_lora(backbone, rank=RANK, alpha=ALPHA)
    after_first = {n for n, p in backbone.named_parameters() if p.requires_grad}

    second = inject_lora(backbone, rank=RANK, alpha=ALPHA)
    after_second = {n for n, p in backbone.named_parameters() if p.requires_grad}

    assert second == [], "second injection should find every target already adapted"
    assert after_second == after_first, (
        f"trainable set changed on re-injection; lost {sorted(after_first - after_second)}"
    )

    # Staged injection: adapting qkv first then proj must leave both trainable.
    staged = _tiny_backbone()
    inject_lora(staged, rank=RANK, alpha=ALPHA, target_modules=["qkv"])
    inject_lora(staged, rank=RANK, alpha=ALPHA, target_modules=["proj"])
    trainable = {n for n, p in staged.named_parameters() if p.requires_grad}
    assert [n for n in trainable if "qkv.lora" in n], "qkv adapters were frozen by the proj pass"
    assert [n for n in trainable if "proj.lora" in n], "proj adapters are not trainable"


@pytest.mark.ssl_unit
def test_unknown_target_module_raises():
    """A typo in target_modules fails loudly rather than silently adapting nothing."""
    with pytest.raises(ValueError, match="Unknown LoRA target_modules"):
        inject_lora(_tiny_backbone(), target_modules=["qkv", "not_a_projection"])


# --------------------------------------------------------------------------------------
# G1.4 — freeze audit (backbone level)
# --------------------------------------------------------------------------------------

@pytest.mark.ssl_unit
def test_backbone_freeze_audit():
    """G1.4: inside the backbone only lora_* and mask_token stay trainable."""
    backbone = _tiny_backbone()
    inject_lora(backbone, rank=RANK, alpha=ALPHA)

    trainable = {n for n, p in backbone.named_parameters() if p.requires_grad}
    expected = {n for n, _ in backbone.named_parameters() if is_lora_key(n)} | {"mask_token"}
    assert trainable == expected, f"unexpected trainable set: {sorted(trainable ^ expected)}"


@pytest.mark.ssl_unit
def test_lora_parameter_report_counts_adapters():
    """The trainable-param report (gate G2.7's log line) counts adapters correctly."""
    backbone = _tiny_backbone()
    inject_lora(backbone, rank=RANK, alpha=ALPHA)
    stats = lora_parameter_report(backbone, name="tiny")

    expected_lora = sum(p.numel() for n, p in backbone.named_parameters() if is_lora_key(n))
    assert stats["lora"] == expected_lora > 0
    assert stats["trainable"] == expected_lora + backbone.mask_token.numel()
    assert 0.0 < stats["trainable_fraction"] < 1.0


# --------------------------------------------------------------------------------------
# Full-model gates (G1.3, G1.4, G1.6, G1.7) — need the PL model, hence CUDA.
# --------------------------------------------------------------------------------------

def _lora_experiment_config(gram=True, preservation=True, backbone="vit_s"):
    """Build a small-but-real DINOv3 experiment config with LoRA enabled."""
    config = OmegaConf.structured(ExperimentConfig())
    config.model.backbone.student_type = backbone
    config.model.backbone.teacher_type = backbone
    config.model.lora.enable = True
    config.model.lora.rank = RANK
    config.model.lora.alpha = ALPHA
    config.model.lora.dropout = 0.0
    config.model.gram.enable = gram
    config.model.gram.w_gram = 1.0 if gram else 0.0
    config.model.preservation.enable = preservation
    return config


@pytest.mark.ssl_unit
def test_ema_zip_alignment():
    """G1.3: student and teacher expose identical parameter names *and shapes* after injection.

    ``update_teacher`` zips ``student.parameters()`` with ``teacher.parameters()``
    element-wise, so any asymmetry in injection silently corrupts the EMA.
    """
    model = DinoV3PlModel(_lora_experiment_config())
    model.inject_lora_adapters()

    student_names = [n for n, _ in model.student.named_parameters()]
    teacher_names = [n for n, _ in model.teacher.named_parameters()]
    assert student_names == teacher_names

    for (name, student_param), (_, teacher_param) in zip(
        model.student.named_parameters(), model.teacher.named_parameters()
    ):
        assert student_param.shape == teacher_param.shape, f"shape mismatch at {name}"

    # The teacher must start *equal* to the student, else its adapter is not an EMA of the
    # student's (gate G2.4 at step 0). lora_A is randomly initialized, so this is a real check.
    student_state = model.student.backbone.state_dict()
    teacher_state = model.teacher.backbone.state_dict()
    for key, value in student_state.items():
        torch.testing.assert_close(teacher_state[key], value, rtol=0, atol=0,
                                   msg=lambda m, k=key: f"teacher != student at {k}: {m}")
    assert any(is_lora_key(k) for k in student_state)


@pytest.mark.ssl_unit
def test_trainable_set():
    """G1.4: across the whole student, only lora_*, the heads and mask_token get gradients."""
    model = DinoV3PlModel(_lora_experiment_config())
    model.inject_lora_adapters()

    for name, param in model.student.named_parameters():
        should_train = (
            is_lora_key(name) or
            name.startswith(("dino_head.", "ibot_head.")) or
            name == "backbone.mask_token"
        )
        assert param.requires_grad == should_train, (
            f"{name}: requires_grad={param.requires_grad}, expected {should_train}"
        )

    # The EMA teacher is never gradient-updated, adapters included.
    assert not any(p.requires_grad for p in model.teacher.parameters())
    # The frozen anchor teacher gets no LoRA at all.
    assert not has_lora(model.gram_teacher)
    assert not any(p.requires_grad for p in model.gram_teacher.parameters())


@pytest.mark.ssl_unit
def test_update_teacher_leaves_frozen_base_bit_exact():
    """The EMA must not touch frozen base weights under LoRA (Stage-2 G2.3 regression).

    ``m*W + (1-m)*W`` is a no-op in exact arithmetic but not in floating point, so the
    inherited all-parameter EMA slowly drifted the teacher's "frozen" base away from the
    student's. Here one update with a realistic momentum must leave every base tensor
    bit-identical while still moving the adapters.
    """
    model = DinoV3PlModel(_lora_experiment_config())
    model.inject_lora_adapters()

    # update_teacher consults trainer.strategy; stand in for a single-device run.
    class _Trainer:
        strategy = SingleDeviceStrategy()
    model.trainer = _Trainer()

    # Move the student's trainable params so the EMA has something real to track.
    with torch.no_grad():
        for name, param in model.student.named_parameters():
            if param.requires_grad:
                param.add_(torch.randn_like(param) * 0.01)

    before = {n: p.detach().clone() for n, p in model.teacher.named_parameters()}
    model.update_teacher(0.9999)

    drifted, moved = [], []
    for name, param in model.teacher.named_parameters():
        changed = not torch.equal(param, before[name])
        if is_lora_key(name) or name.startswith(("dino_head.", "ibot_head.")) or \
                name == "backbone.mask_token":
            if changed:
                moved.append(name)
        elif changed:
            drifted.append(name)

    assert not drifted, f"frozen base tensors changed during EMA: {drifted[:5]}"
    assert moved, "EMA did not update any trainable parameter"


@pytest.mark.ssl_unit
def test_gram_sync_lora():
    """G1.6: syncing a LoRA-injected teacher into the un-injected anchor teacher works."""
    model = DinoV3PlModel(_lora_experiment_config())
    model.inject_lora_adapters()

    # Perturb the teacher's base weights so a successful sync is observable.
    with torch.no_grad():
        model.teacher.backbone.cls_token.add_(1.0)
    _randomize_lora_(model.teacher.backbone)

    model._sync_gram_teacher()  # must not raise on the lora_* keys

    torch.testing.assert_close(
        model.gram_teacher.cls_token, model.teacher.backbone.cls_token, rtol=0, atol=0
    )
    assert not has_lora(model.gram_teacher)
    assert set(model.gram_teacher.state_dict()) == set(
        strip_lora_keys(model.teacher.backbone.state_dict())
    )


@pytest.mark.ssl_unit
def test_optim_groups():
    """G1.7: LoRA params land in the correct blocks.N layer-id groups; no frozen params appear."""
    model = DinoV3PlModel(_lora_experiment_config())
    model.inject_lora_adapters()

    optimizer = model.configure_optimizers()
    grouped = {id(p) for group in optimizer.param_groups for p in group["params"]}

    trainable = {id(p) for p in model.student.parameters() if p.requires_grad}
    frozen = {id(p) for p in model.student.parameters() if not p.requires_grad}

    assert grouped == trainable, "optimizer groups do not match the trainable set"
    assert not (grouped & frozen), "frozen params leaked into an optimizer group"

    # Layer-id decay: a LoRA param in block N must get the same lr_multiplier as that block's
    # base weights would, i.e. the `int(after_block[0])` fallback in configure_optimizers fires.
    n_blocks = model.student.backbone.n_blocks
    multiplier_by_lora_param = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            multiplier_by_lora_param[id(param)] = group["lr_multiplier"]

    for name, param in model.student.named_parameters():
        if not is_lora_key(name):
            continue
        block_index = int(name.split(".blocks.")[-1].split(".")[0])
        expected = model.layerwise_decay ** (n_blocks + 1 - (block_index + 1))
        assert multiplier_by_lora_param[id(param)] == pytest.approx(expected), (
            f"{name} landed in the wrong layer-id group"
        )


# --------------------------------------------------------------------------------------
# Config plumbing and guardrails
# --------------------------------------------------------------------------------------

@pytest.mark.ssl_unit
def test_lora_and_preservation_config_defaults():
    """The new config fields exist with the documented defaults."""
    config = OmegaConf.structured(ExperimentConfig())

    assert config.model.lora.enable is False
    assert config.model.lora.rank == 8
    assert config.model.lora.alpha == 16.0
    assert config.model.lora.dropout == 0.05
    assert list(config.model.lora.target_modules) == ["qkv", "proj"]
    assert config.model.lora.num_last_blocks == 0

    assert config.model.preservation.enable is False
    assert config.model.preservation.cls_mse_weight == 0.05
    assert config.model.preservation.cls_cosine_weight == 0.05


@pytest.mark.ssl_unit
def test_anchor_teacher_builds_for_preservation_without_gram():
    """The anchor teacher must exist when preservation alone is on (gram disabled)."""
    model = DinoV3PlModel(_lora_experiment_config(gram=False, preservation=True))
    assert getattr(model, "gram_teacher", None) is not None
    assert not has_lora(model.gram_teacher)


@pytest.mark.ssl_unit
def test_no_anchor_teacher_when_both_disabled():
    """With neither gram nor preservation, no anchor teacher is built (memory unchanged)."""
    config = _lora_experiment_config(gram=False, preservation=False)
    model = DinoV3PlModel(config)
    assert getattr(model, "gram_teacher", None) is None
    assert not model._extra_losses()


@pytest.mark.ssl_unit
@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda c: setattr(c.model.distill, "enable", True),
         r"model\.lora\.enable is not supported together with model\.distill\.enable"),
        (lambda c: setattr(c.model.gram, "teacher_source", "ema"),
         r"model\.lora\.enable requires model\.gram\.teacher_source='pretrained'"),
        (lambda c: setattr(c.train, "distributed_strategy", "fsdp"),
         r"model\.lora\.enable is not supported with train\.distributed_strategy='fsdp'"),
    ],
    ids=["distill", "gram_ema", "fsdp"],
)
def test_lora_guardrails(mutate, expected):
    """Each v1 incompatibility is rejected at build time with *its own* actionable message.

    The messages are matched exactly, not by keyword: the distillation case in particular
    would otherwise be satisfied by the inherited ``_build_model`` assert about a missing
    ``pretrained_non_distill_pl_model_path``, which fires later and for a different reason.
    """
    config = _lora_experiment_config()
    mutate(config)
    with pytest.raises(AssertionError, match=expected):
        DinoV3PlModel(config)


@pytest.mark.ssl_unit
@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: setattr(c.model.distill, "enable", True),
        lambda c: setattr(c.model.gram, "teacher_source", "ema"),
        lambda c: setattr(c.train, "distributed_strategy", "fsdp"),
    ],
    ids=["distill", "gram_ema", "fsdp"],
)
def test_guardrails_do_not_fire_when_lora_is_disabled(mutate):
    """The guardrails constrain LoRA only — they must not restrict existing full-FT configs."""
    config = _lora_experiment_config()
    config.model.lora.enable = False
    mutate(config)
    # Any failure here must not come from _validate_lora_config.
    try:
        DinoV3PlModel(config)
    except AssertionError as error:
        assert "model.lora.enable" not in str(error), (
            f"LoRA guardrail fired with lora disabled: {error}"
        )


# --------------------------------------------------------------------------------------
# Preservation loss behaviour (the G2.1 zero-start property, checked statically)
# --------------------------------------------------------------------------------------

@pytest.mark.ssl_unit
def test_cls_preservation_loss_primitive_zero_for_identical_inputs():
    """Both CLS preservation terms are exactly 0 when student == anchor (gate G2.1 at step 0)."""
    from nvidia_tao_pytorch.ssl.dinov3.model.loss import ClsPreservationLoss

    torch.manual_seed(0)
    cls_tokens = torch.randn(BATCH_SIZE, 64)
    cls_mse, cls_cosine = ClsPreservationLoss()(cls_tokens, cls_tokens.clone())

    assert cls_mse.item() == pytest.approx(0.0, abs=1e-7)
    assert cls_cosine.item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.ssl_unit
def test_extra_losses_identity_through_the_real_path():
    """Exercise ``_extra_losses`` itself, with the masking contract made explicit.

    ``loss_fn(x, x) == 0`` only tests the primitive. Here the student and anchor are the same
    weights by construction (LoRA is an identity at ``lora_B = 0``), and the terms are driven
    through the production ``_extra_losses`` path. With masking and stochastic depth disabled
    the identity is exact; with masking on, the terms are legitimately far from zero because
    the student sees masked crops and the anchor does not. Both are asserted so a real
    sync/remap regression cannot hide behind the masked value.
    """
    config = _lora_experiment_config()
    config.model.backbone.drop_path_rate = 0.0
    # fp32 identity check, so the xformers path (which hard-casts q/k/v to .half()) must be off.
    config.train.use_custom_attention = False
    model = DinoV3PlModel(config)
    model.inject_lora_adapters()
    model.eval()

    size, patch = 64, model.patch_size
    n_tokens = (size // patch) ** 2
    torch.manual_seed(0)
    crops = torch.randn(2, 3, size, size)

    captured = {}
    model._log_loss = lambda name, value: captured.__setitem__(name, float(value))

    def run(masks):
        with torch.no_grad():
            out = model.student.backbone([crops, crops], masks=[masks, None])[0]
        captured.clear()
        model._extra_losses(student_backbone_global_output=out, global_crops=crops)
        return dict(captured)

    unmasked = run(torch.zeros(2, n_tokens, dtype=torch.bool))
    assert unmasked["losses/cls_mse"] < 1e-5, unmasked
    assert unmasked["losses/cls_cos"] < 1e-5, unmasked
    assert unmasked["losses/gram_loss"] < 1e-5, unmasked

    masked = run(torch.rand(2, n_tokens) < 0.3)
    assert masked["losses/cls_cos"] > unmasked["losses/cls_cos"], (
        "masking must move the CLS term; if it does not, the anchor is seeing the same "
        "masked input and the documented contract is wrong"
    )


@pytest.mark.ssl_unit
def test_cls_preservation_loss_grows_with_drift():
    """Both terms increase monotonically as the student drifts away from the anchor."""
    from nvidia_tao_pytorch.ssl.dinov3.model.loss import ClsPreservationLoss

    torch.manual_seed(0)
    anchor = torch.randn(BATCH_SIZE, 64)
    noise = torch.randn_like(anchor)
    loss_fn = ClsPreservationLoss()

    previous = (-1.0, -1.0)
    for scale in (0.1, 0.5, 1.0):
        cls_mse, cls_cosine = loss_fn(anchor + scale * noise, anchor)
        assert cls_mse.item() > previous[0]
        assert cls_cosine.item() > previous[1]
        previous = (cls_mse.item(), cls_cosine.item())
