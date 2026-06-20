from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from time import strftime
from typing import Any

from model.method.contrastive_lora.main import ContrastiveLora


class SubRiemannianLora(ContrastiveLora):
    name = "sr_lora"
    needs_hallucinations = True

    def __init__(self, env_path: str = "env.sh") -> None:
        super().__init__(env_path)
        self.include_mlp = self.bool_value(self.env.get("SR_INCLUDE_MLP", "true"))
        self.include_attn = self.bool_value(self.env.get("SR_INCLUDE_ATTN", "false"))
        self.sr_target_modules = self.parse_csv(self.env.get("SR_TARGET_MODULES", "gate_proj,up_proj,down_proj"))
        self.target_layers = self.parse_target_layers(self.env.get("SR_TARGET_LAYERS", "1"))
        self.target_layer_indices = self.parse_layer_indices(self.env.get("SR_TARGET_LAYER_INDICES", "none"))
        self.finite_diff_eps = float(self.env.get("SR_FINITE_DIFF_EPS", "1e-8"))
        self.x_clip = float(self.env.get("SR_X_CLIP", "10.0"))
        self.metric_gain = float(self.env.get("SR_METRIC_GAIN", "8.0"))
        self.metric_min = float(self.env.get("SR_METRIC_MIN", "1.0"))
        self.metric_max = float(self.env.get("SR_METRIC_MAX", "8.0"))
        self.metric_temp = float(self.env.get("SR_METRIC_TEMPERATURE", "1.0"))
        self.condition_max = float(self.env.get("SR_METRIC_COND_MAX", "16.0"))
        self.lambda_fit = float(self.env.get("SR_LAMBDA_FIT", "1.0"))
        self.lambda_identity = float(self.env.get("SR_LAMBDA_IDENTITY", "0.01"))
        self.lambda_bound = float(self.env.get("SR_LAMBDA_BOUND", "0.01"))
        self.lambda_condition = float(self.env.get("SR_LAMBDA_CONDITION", "0.01"))
        self.lambda_lookahead = float(self.env.get("SR_LAMBDA_LOOKAHEAD", "1.0"))
        self.lookahead_enabled = self.bool_value(self.env.get("SR_LOOKAHEAD_ENABLED", "true"))
        self.lookahead_lr = float(self.env.get("SR_LOOKAHEAD_LR", self.env.get("TRAIN_LR", "1e-5")))
        self.lookahead_margin = float(self.env.get("SR_LOOKAHEAD_MARGIN", "0.0"))
        self.lookahead_every = max(1, int(self.env.get("SR_LOOKAHEAD_EVERY", "1")))
        self.lookahead_max_samples = max(1, int(self.env.get("SR_LOOKAHEAD_MAX_SAMPLES", "128")))
        self.lookahead_chunk_size = max(1, int(self.env.get("SR_LOOKAHEAD_CHUNK_SIZE", "2")))
        self._current_contrast_grads: dict[int, Any] = {}
        self._previous_boundary: dict[int, dict[str, Any]] = {}
        self._lookahead_batches: list[tuple[Any, tuple[Any, ...]]] = []
        self._last_step_rows: list[dict[str, Any]] = []
        self.visualization_enabled = self.bool_value(self.env.get("SR_VISUALIZE_MASK", "true"))
        self.visualization_interval = max(1, int(self.env.get("SR_VISUALIZE_EVERY", self.env.get("EVAL_EVERY_STEPS", "50"))))
        self.visualization_max_elements = max(1, int(self.env.get("SR_VISUALIZE_MAX_ELEMENTS", "4096")))
        self.visualization_dir = Path(
            self.env.get("SR_VISUALIZATION_DIR", str(Path(self.env.get("OUTPUT_ROOT", "outputs")) / "visualization" / "soft_mask"))
        )
        self.visualization_summary_path = self.visualization_dir / "soft_mask_summary.jsonl"
        self.visualization_manifest_path = self.visualization_dir / "manifest.json"
        self.visualization_run_id = strftime("%Y%m%d_%H%M%S")
        # Visualization is checked at optimizer-step time, not every raw train step.
        # With grad accumulation, optimizer steps may be 16,32,48,64,... while
        # SR_VISUALIZE_EVERY is often 50. A strict step % 50 == 0 check can
        # therefore record no arrays at all. Use threshold scheduling instead.
        self._next_visualization_step = self.visualization_interval
        self.prepare_visualization_dir()
        self.debug = self.build_debug_recorder()

    def apply(self, model: Any) -> Any:
        model = super().apply(model)
        self.attach_sr_metric_parameters(model)
        return model

    def loss(self, tokenizer: Any, max_length: int | None) -> Any:
        from model.method.sr_lora.loss import SubRiemannianLoraLoss

        return SubRiemannianLoraLoss(tokenizer=tokenizer, max_length=max_length)

    def backward(self, model: Any, out: Any, grad_accum_steps: int, step: int) -> None:
        if out.hall_ce is None:
            (out.loss / grad_accum_steps).backward()
            return

        self.remember_lookahead_batch(out)

        trainable = [param for param in model.parameters() if param.requires_grad]
        saved_grads = [(param, None if param.grad is None else param.grad.detach().clone()) for param in trainable]

        self.clear_grads(trainable)
        (out.gold_ce / grad_accum_steps).backward(retain_graph=True)
        gold_grads = [(param, None if param.grad is None else param.grad.detach().clone()) for param in trainable]

        self.clear_grads(trainable)
        (out.hall_ce / grad_accum_steps).backward()
        hall_grads = [(param, None if param.grad is None else param.grad.detach().clone()) for param in trainable]

        self.clear_grads(trainable)
        target_by_param_id = {id(module.weight): (name, module) for name, module in self.target_lora_b_modules(model)}
        for (param, saved), (_, gold), (_, hall) in zip(saved_grads, gold_grads, hall_grads):
            if gold is None and hall is None:
                param.grad = saved
                continue

            if gold is None:
                combined = -hall
            elif hall is None:
                combined = gold
            else:
                combined = gold - hall

            param.grad = combined if saved is None else saved + combined
            target = target_by_param_id.get(id(param))
            if target is not None:
                _, module = target
                module_id = id(module)
                current = self._current_contrast_grads.get(module_id)
                detached = combined.detach().float()
                self._current_contrast_grads[module_id] = detached.clone() if current is None else current + detached

    def before_optimizer_step(self, model: Any, step: int) -> dict[str, Any] | None:
        targets = self.target_lora_b_modules(model)
        if not targets:
            self._current_contrast_grads.clear()
            self._lookahead_batches.clear()
            return {"active": False, "reason": "no_sr_target_lora_b"}

        rows = []
        for name, module in targets:
            row = self.precondition_module_gradient(model, step, name, module)
            if row is not None:
                rows.append(row)

        self._current_contrast_grads.clear()
        self._lookahead_batches.clear()
        self._last_step_rows = rows
        if self.debug is not None:
            self.debug.record_step(step=step, rows=rows)

        active_rows = [row for row in rows if row.get("active")]
        if not active_rows:
            reasons = sorted({str(row.get("reason")) for row in rows}) if rows else ["no_sr_grad"]
            return {"active": False, "reason": ",".join(reasons), "modules": len(rows)}

        return {
            "active": True,
            "modules": len(active_rows),
            "skipped_modules": len(rows) - len(active_rows),
            "avg_suppression_ratio": self.mean(row["suppression_ratio"] for row in active_rows),
            "avg_small_denom_fraction": self.mean(row["small_denom_fraction"] for row in active_rows),
            "avg_metric_loss": self.mean(row["metric_loss"] for row in active_rows),
            "avg_lookahead_objective_gain": self.mean(
                row.get("lookahead", {}).get("objective_gain", 0.0) for row in active_rows
            ),
            "lookahead_better_fraction": self.mean(
                1.0 if row.get("lookahead", {}).get("masked_better_than_plain") else 0.0
                for row in active_rows
            ),
            "metric_min": min(row["metric_min"] for row in active_rows),
            "metric_max": max(row["metric_max"] for row in active_rows),
            "sample_modules": active_rows[:3],
        }

    def precondition_module_gradient(self, model: Any, step: int, name: str, module: Any) -> dict[str, Any] | None:
        import torch

        module_id = id(module)
        current_grad = self._current_contrast_grads.get(module_id)
        if current_grad is None:
            return {"active": False, "name": name, "reason": "no_current_grad"}

        weight = module.weight
        current_weight = weight.detach().float().clone()
        previous = self._previous_boundary.get(module_id)
        if previous is None:
            self._previous_boundary[module_id] = {
                "weight": current_weight,
                "contrast_grad": current_grad.detach().float().clone(),
            }
            return {"active": False, "name": name, "reason": "first_boundary_skip"}

        delta_weight = current_weight - previous["weight"].to(device=current_weight.device)
        delta_contrast = current_grad.detach().float() - previous["contrast_grad"].to(device=current_grad.device)
        x_raw, valid = self.secant_ratio(delta_contrast, delta_weight)
        x_clipped = self.signed_soft_clip(x_raw, self.x_clip)
        x_unit = self.l2_normalize_signed(x_clipped).detach()
        metric_diag = self.metric_for_update(module.sr_metric_raw, x_unit)
        inv_metric = metric_diag.reciprocal()

        if weight.grad is not None:
            preconditioned = (weight.grad.detach().float() * inv_metric).to(dtype=weight.grad.dtype)
            weight.grad.copy_(preconditioned)

        param_name = f"{name}.weight"
        metric_loss, loss_parts, lookahead_row = self.metric_parameter_loss(
            model=model,
            param_name=param_name,
            module=module,
            x_unit=x_unit,
            contrast_grad=current_grad,
            step=step,
        )
        metric_grad = torch.autograd.grad(metric_loss, module.sr_metric_raw, allow_unused=True)[0]
        if metric_grad is not None:
            module.sr_metric_raw.grad = metric_grad.detach()

        raw_grad_norm = self.norm(current_grad)
        preconditioned_norm = self.norm(current_grad * inv_metric)
        mask_stats = self.tensor_distribution(inv_metric)
        self.record_soft_mask_visualization(
            step=step,
            name=name,
            mask=inv_metric,
            metric=metric_diag,
            x_unit=x_unit,
        )

        row = {
            "active": True,
            "name": name,
            "shape": list(weight.shape),
            "soft_mask": mask_stats,
            "d_weight_norm": self.norm(delta_weight),
            "d_contrast_grad_norm": self.norm(delta_contrast),
            "x_norm": self.norm(x_unit),
            "x_min": self.scalar(x_unit.min()),
            "x_max": self.scalar(x_unit.max()),
            "x_mean": self.scalar(x_unit.mean()),
            "x_negative_fraction": self.scalar((x_unit < 0).float().mean()),
            "x_positive_fraction": self.scalar((x_unit > 0).float().mean()),
            "small_denom_fraction": 1.0 - self.scalar(valid.float().mean()),
            "metric_min": self.scalar(metric_diag.min()),
            "metric_max": self.scalar(metric_diag.max()),
            "metric_mean": self.scalar(metric_diag.mean()),
            "inv_metric_mean": self.scalar(inv_metric.mean()),
            "raw_grad_norm": raw_grad_norm,
            "preconditioned_grad_norm": preconditioned_norm,
            "suppression_ratio": self.safe_ratio(preconditioned_norm, raw_grad_norm),
            "metric_loss": self.scalar(metric_loss),
            "metric_loss_parts": loss_parts,
            "lookahead": lookahead_row,
        }
        self._previous_boundary[module_id] = {
            "weight": current_weight,
            "contrast_grad": current_grad.detach().float().clone(),
        }
        return row

    def attach_sr_metric_parameters(self, model: Any) -> None:
        import torch

        targets = self.target_lora_b_modules(model)
        if not targets:
            print("[sr_lora] no elementwise lora_B targets found; SR preconditioner disabled", flush=True)
            return

        for name, module in targets:
            shape = tuple(module.weight.shape)
            current = getattr(module, "sr_metric_raw", None)
            if current is not None and tuple(current.shape) != shape:
                del module._parameters["sr_metric_raw"]
                current = None
            if current is None:
                module.register_parameter(
                    "sr_metric_raw",
                    torch.nn.Parameter(torch.zeros(shape, device=module.weight.device, dtype=torch.float32)),
                )
            print(f"[sr_lora] attached elementwise metric to {name} shape={shape}", flush=True)

        print(
            "[sr_lora] active elementwise metrics="
            f"{len(targets)} modules={self.target_modules_label()} layers={self.target_layers_label()} "
            f"layer_indices={self.target_layer_indices_label()} include_mlp={self.include_mlp} "
            f"include_attn={self.include_attn}",
            flush=True,
        )

    def target_lora_b_modules(self, model: Any) -> list[tuple[str, Any]]:
        candidates = [
            (name, module)
            for name, module in model.named_modules()
            if "lora_B" in name and hasattr(module, "weight") and self.matches_target(name)
        ]
        candidates = self.filter_layer_indices(candidates)
        if self.target_layers is None:
            return candidates

        selected = []
        seen: set[int] = set()
        for target_name in self.target_module_groups(candidates):
            group = [(name, module) for name, module in candidates if self.module_group_name(name) == target_name]
            for name, module in group[-self.target_layers :]:
                module_id = id(module)
                if module_id not in seen:
                    selected.append((name, module))
                    seen.add(module_id)
        return selected

    def matches_target(self, name: str) -> bool:
        lowered = name.lower()
        is_mlp = any(token in lowered for token in ("gate_proj", "up_proj", "down_proj", ".mlp."))
        is_attn = any(token in lowered for token in ("q_proj", "k_proj", "v_proj", "o_proj", "self_attn", ".attn."))
        if is_mlp and not self.include_mlp:
            return False
        if is_attn and not self.include_attn:
            return False
        if not is_mlp and not is_attn and not (self.include_mlp and self.include_attn):
            return False
        if self.sr_target_modules == ["all"]:
            return True
        return any(token in name for token in self.sr_target_modules)

    def filter_layer_indices(self, candidates: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
        if self.target_layer_indices is None:
            return candidates
        return [
            (name, module)
            for name, module in candidates
            if self.layer_index(name) in self.target_layer_indices
        ]

    def target_module_groups(self, candidates: list[tuple[str, Any]]) -> list[str]:
        groups = []
        seen = set()
        for name, _ in candidates:
            group = self.module_group_name(name)
            if group not in seen:
                groups.append(group)
                seen.add(group)
        return groups

    @staticmethod
    def module_group_name(name: str) -> str:
        for token in ("gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"):
            if token in name:
                return token
        return "other"

    @staticmethod
    def layer_index(name: str) -> int | None:
        match = re.search(r"(?:layers|h|blocks)\.(\d+)", name)
        return None if match is None else int(match.group(1))

    def secant_ratio(self, delta_grad: Any, delta_weight: Any) -> tuple[Any, Any]:
        import torch

        valid = delta_weight.abs() >= self.finite_diff_eps
        signed_denom = delta_weight.sign() * delta_weight.abs().clamp_min(self.finite_diff_eps)
        ratio = delta_grad / signed_denom
        ratio = torch.where(valid & torch.isfinite(ratio), ratio, torch.zeros_like(ratio))
        return ratio, valid

    @staticmethod
    def signed_soft_clip(values: Any, clip: float) -> Any:
        import torch

        clean = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
        if clip <= 0:
            return clean
        return clip * torch.tanh(clean / clip)

    def metric_from_x(self, x_unit: Any) -> Any:
        import torch

        risk = x_unit.abs()
        metric = 1.0 + self.metric_gain * risk
        return torch.clamp(metric, min=self.metric_min, max=self.metric_max)

    def metric_for_update(self, metric_raw: Any, x_unit: Any) -> Any:
        raw_signal = self.raw_metric_signal(metric_raw.detach())
        if self.norm(raw_signal) <= 1e-12:
            return self.metric_from_x(x_unit).detach()
        return self.metric_from_signal(self.l2_normalize_signed(raw_signal)).detach()

    def metric_from_raw(self, metric_raw: Any) -> Any:
        return self.metric_from_signal(self.l2_normalize_signed(self.raw_metric_signal(metric_raw)))

    def metric_from_signal(self, signal: Any) -> Any:
        import torch

        metric = 1.0 + self.metric_gain * signal.abs()
        return torch.clamp(metric, min=self.metric_min, max=self.metric_max)

    def raw_metric_signal(self, metric_raw: Any) -> Any:
        import torch

        return torch.tanh(metric_raw.float() / max(self.metric_temp, 1e-12))

    def metric_parameter_loss(
        self,
        model: Any,
        param_name: str,
        module: Any,
        x_unit: Any,
        contrast_grad: Any,
        step: int,
    ) -> tuple[Any, dict[str, float], dict[str, Any]]:
        import torch
        import torch.nn.functional as F

        metric_raw = module.sr_metric_raw
        raw_signal = self.raw_metric_signal(metric_raw)
        learned_unit = self.l2_normalize_signed(raw_signal)
        unbounded_metric = 1.0 + self.metric_gain * learned_unit.abs()
        fit_loss = (learned_unit - x_unit).pow(2).mean()
        identity_loss = (unbounded_metric - 1.0).pow(2).mean()
        bound_loss = F.relu(self.metric_min - unbounded_metric).pow(2).mean()
        bound_loss = bound_loss + F.relu(unbounded_metric - self.metric_max).pow(2).mean()
        condition = unbounded_metric.max() / unbounded_metric.min().clamp_min(1e-12)
        condition_loss = F.relu(condition - self.condition_max).pow(2)
        lookahead_loss, lookahead_row = self.lookahead_improvement_loss(
            model=model,
            param_name=param_name,
            module=module,
            contrast_grad=contrast_grad,
            step=step,
        )
        loss = (
            self.lambda_fit * fit_loss
            + self.lambda_identity * identity_loss
            + self.lambda_bound * bound_loss
            + self.lambda_condition * condition_loss
            + self.lambda_lookahead * lookahead_loss
        )
        return loss, {
            "fit": self.scalar(fit_loss),
            "identity": self.scalar(identity_loss),
            "bound": self.scalar(bound_loss),
            "condition": self.scalar(condition_loss),
            "condition_value": self.scalar(condition),
            "lookahead": self.scalar(lookahead_loss),
        }, lookahead_row

    def remember_lookahead_batch(self, out: Any) -> None:
        if not self.lookahead_enabled:
            return
        loss_fn = getattr(out, "loss_fn", None)
        samples = getattr(out, "samples", None)
        if loss_fn is None or samples is None:
            return
        self._lookahead_batches.append((loss_fn, tuple(samples)))

    def lookahead_improvement_loss(
        self,
        model: Any,
        param_name: str,
        module: Any,
        contrast_grad: Any,
        step: int,
    ) -> tuple[Any, dict[str, Any]]:
        import torch
        import torch.nn.functional as F

        zero = module.sr_metric_raw.new_tensor(0.0)
        if not self.lookahead_enabled:
            return zero, {"active": False, "reason": "disabled"}
        if step % self.lookahead_every != 0:
            return zero, {"active": False, "reason": "interval"}
        probe = self.lookahead_probe_batch()
        if probe is None:
            return zero, {"active": False, "reason": "no_probe_batch"}

        loss_fn, samples = probe
        weight = module.weight
        update = contrast_grad.detach().to(device=weight.device, dtype=torch.float32)
        metric_diag = self.metric_from_raw(module.sr_metric_raw)
        masked_update = update * metric_diag.reciprocal()
        plain_param = weight - self.lookahead_lr * update.to(device=weight.device, dtype=weight.dtype)
        masked_param = weight - self.lookahead_lr * masked_update.to(device=weight.device, dtype=weight.dtype)

        with torch.no_grad():
            plain_objective, plain_parts = self.lookahead_objective(
                model=model,
                loss_fn=loss_fn,
                samples=samples,
                param_name=param_name,
                updated_param=plain_param,
            )
        masked_objective, masked_parts = self.lookahead_objective(
            model=model,
            loss_fn=loss_fn,
            samples=samples,
            param_name=param_name,
            updated_param=masked_param,
        )
        objective_delta = masked_objective - plain_objective.detach()
        loss = F.softplus(objective_delta + self.lookahead_margin)
        gain = plain_objective.detach() - masked_objective.detach()
        return loss, {
            "active": True,
            "probe_samples": len(samples),
            "chunk_size": self.lookahead_chunk_size,
            "plain_gold_ce": self.float_or_none(plain_parts["gold_ce"]),
            "masked_gold_ce": self.float_or_none(masked_parts["gold_ce"]),
            "plain_hall_ce": self.float_or_none(plain_parts["hall_ce"]),
            "masked_hall_ce": self.float_or_none(masked_parts["hall_ce"]),
            "plain_objective": self.scalar(plain_objective),
            "masked_objective": self.scalar(masked_objective.detach()),
            "objective_gain": self.scalar(gain),
            "masked_better_than_plain": bool(self.scalar(gain) > 0.0),
            "loss": self.scalar(loss.detach()),
            "margin": self.lookahead_margin,
        }

    def lookahead_probe_batch(self) -> tuple[Any, tuple[Any, ...]] | None:
        if not self._lookahead_batches:
            return None
        loss_fn, samples = self._lookahead_batches[-1]
        selected = tuple(samples[: self.lookahead_max_samples])
        return None if not selected else (loss_fn, selected)

    def lookahead_objective(
        self,
        model: Any,
        loss_fn: Any,
        samples: Sequence[Any],
        param_name: str,
        updated_param: Any,
    ) -> tuple[Any, dict[str, Any]]:
        gold_ce = self.lookahead_sequence_ce(
            model=model,
            loss_fn=loss_fn,
            samples=samples,
            answer_attr="answer",
            param_name=param_name,
            updated_param=updated_param,
            chunk_size=self.lookahead_chunk_size,
        )
        hall_samples = tuple(sample for sample in samples if getattr(sample, "hallucinated_answer", ""))
        hall_ce = None
        objective = gold_ce
        if hall_samples:
            hall_ce = self.lookahead_sequence_ce(
                model=model,
                loss_fn=loss_fn,
                samples=hall_samples,
                answer_attr="hallucinated_answer",
                param_name=param_name,
                updated_param=updated_param,
                chunk_size=self.lookahead_chunk_size,
            )
            objective = objective - hall_ce
        return objective, {"gold_ce": gold_ce, "hall_ce": hall_ce}

    @staticmethod
    def lookahead_sequence_ce(
        model: Any,
        loss_fn: Any,
        samples: Sequence[Any],
        answer_attr: str,
        param_name: str,
        updated_param: Any,
        chunk_size: int = 2,
    ) -> Any:
        import torch.nn.functional as F
        from torch.func import functional_call

        total_loss = updated_param.new_tensor(0.0, dtype=updated_param.float().dtype)
        total_tokens = updated_param.new_tensor(0.0, dtype=updated_param.float().dtype)
        chunk_size = max(1, int(chunk_size))
        for start in range(0, len(samples), chunk_size):
            chunk = samples[start : start + chunk_size]
            if not chunk:
                continue
            batch = loss_fn.build_lm_batch(chunk, answer_attr)
            batch = {key: value.to(updated_param.device) for key, value in batch.items()}
            logits = functional_call(
                model,
                {param_name: updated_param},
                (),
                {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
            ).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            total_loss = total_loss + F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_tokens = total_tokens + shift_labels.ne(-100).sum().to(device=total_tokens.device, dtype=total_tokens.dtype)
        return total_loss / total_tokens.clamp_min(1.0)

    def prepare_visualization_dir(self) -> None:
        if not self.visualization_enabled:
            return
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.write_json_file(
            self.visualization_manifest_path,
            {
                "method": self.name,
                "created_at": self.visualization_run_id,
                "run_name": self.env.get("SR_RUN_NAME"),
                "run_group": self.env.get("SR_RUN_GROUP"),
                "ablation": {
                    "method": self.env.get("SR_ABLATION_METHOD", self.env.get("EXPERIMENTS")),
                    "layer_name": self.env.get("SR_ABLATION_LAYER_NAME"),
                    "module_name": self.env.get("SR_ABLATION_MODULE_NAME"),
                    "output_root": self.env.get("SR_ABLATION_OUTPUT_ROOT", self.env.get("OUTPUT_ROOT")),
                    "visualization_dir": self.env.get("SR_ABLATION_VIS_DIR", str(self.visualization_dir)),
                },
                "description": "SR-LoRA soft mask visualization data. soft_mask = metric^{-1}; lower values mean stronger suppression.",
                "files": {
                    "summary": str(self.visualization_summary_path),
                    "heatmaps": str(self.visualization_dir / "heatmaps"),
                    "arrays": str(self.visualization_dir / "arrays"),
                },
                "env": {
                    "OUTPUT_ROOT": self.env.get("OUTPUT_ROOT"),
                    "SR_VISUALIZE_MASK": self.env.get("SR_VISUALIZE_MASK"),
                    "SR_VISUALIZE_EVERY": self.env.get("SR_VISUALIZE_EVERY"),
                    "SR_VISUALIZE_MAX_ELEMENTS": self.env.get("SR_VISUALIZE_MAX_ELEMENTS"),
                    "SR_VISUALIZATION_DIR": self.env.get("SR_VISUALIZATION_DIR"),
                    "SR_TARGET_MODULES": self.env.get("SR_TARGET_MODULES"),
                    "SR_TARGET_LAYERS": self.env.get("SR_TARGET_LAYERS"),
                    "SR_TARGET_LAYER_INDICES": self.env.get("SR_TARGET_LAYER_INDICES"),
                    "SR_METRIC_MIN": self.env.get("SR_METRIC_MIN"),
                    "SR_METRIC_MAX": self.env.get("SR_METRIC_MAX"),
                    "SR_METRIC_GAIN": self.env.get("SR_METRIC_GAIN"),
                },
            },
        )

    def record_soft_mask_visualization(self, *, step: int, name: str, mask: Any, metric: Any, x_unit: Any) -> None:
        if not self.visualization_enabled:
            return
        if step < self._next_visualization_step:
            return
        while self._next_visualization_step <= step:
            self._next_visualization_step += self.visualization_interval
        import numpy as np

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        arrays_dir = self.visualization_dir / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)
        file_stem = f"step_{int(step):06d}__{safe_name}"
        array_path = arrays_dir / f"{file_stem}.npz"
        mask_cpu = mask.detach().float().cpu()
        metric_cpu = metric.detach().float().cpu()
        x_cpu = x_unit.detach().float().cpu()
        np.savez_compressed(
            array_path,
            soft_mask=mask_cpu.numpy(),
            metric=metric_cpu.numpy(),
            x_unit=x_cpu.numpy(),
        )
        row = {
            "step": int(step),
            "module": name,
            "shape": list(mask_cpu.shape),
            "array_path": str(array_path),
            "soft_mask": self.tensor_distribution(mask_cpu),
            "metric": self.tensor_distribution(metric_cpu),
            "x_unit": self.tensor_distribution(x_cpu),
            "interpretation": "soft_mask is metric^{-1}; smaller coordinates are more strongly suppressed.",
        }
        self.append_jsonl_file(self.visualization_summary_path, row)

    @staticmethod
    def tensor_distribution(tensor: Any) -> dict[str, Any]:
        import torch

        flat = tensor.detach().float().reshape(-1)
        if flat.numel() == 0:
            return {"numel": 0}
        clean = torch.where(torch.isfinite(flat), flat, torch.zeros_like(flat))
        quantiles = torch.quantile(clean.cpu(), torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
        return {
            "numel": int(clean.numel()),
            "mean": float(clean.mean().cpu()),
            "std": float(clean.std(unbiased=False).cpu()),
            "min": float(clean.min().cpu()),
            "max": float(clean.max().cpu()),
            "p01": float(quantiles[0]),
            "p05": float(quantiles[1]),
            "p25": float(quantiles[2]),
            "p50": float(quantiles[3]),
            "p75": float(quantiles[4]),
            "p95": float(quantiles[5]),
            "p99": float(quantiles[6]),
            "fraction_below_0_99": float((clean < 0.99).float().mean().cpu()),
            "fraction_below_0_95": float((clean < 0.95).float().mean().cpu()),
            "fraction_below_0_90": float((clean < 0.90).float().mean().cpu()),
            "fraction_below_0_75": float((clean < 0.75).float().mean().cpu()),
            "fraction_below_0_50": float((clean < 0.50).float().mean().cpu()),
        }

    @staticmethod
    def write_json_file(path: Path, data: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def append_jsonl_file(path: Path, row: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def l2_normalize_signed(values: Any) -> Any:
        import torch

        clean = torch.where(torch.isfinite(values), values, torch.zeros_like(values)).float()
        norm = clean.norm()
        if float(norm.detach().cpu()) <= 1e-12:
            return clean
        return clean / norm

    @staticmethod
    def clear_grads(parameters: list[Any]) -> None:
        for param in parameters:
            param.grad = None

    @staticmethod
    def norm(values: Any) -> float:
        return float(values.detach().float().norm().cpu())

    @staticmethod
    def scalar(value: Any) -> float:
        return float(value.detach().float().cpu())

    @classmethod
    def float_or_none(cls, value: Any | None) -> float | None:
        return None if value is None else cls.scalar(value)

    @staticmethod
    def safe_ratio(num: float, den: float) -> float:
        return 0.0 if abs(den) < 1e-30 else num / den

    @staticmethod
    def mean(values: Any) -> float:
        items = list(values)
        return sum(items) / len(items) if items else 0.0

    @staticmethod
    def bool_value(value: str) -> bool:
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"expected bool value, got {value}")

    @staticmethod
    def parse_csv(value: str) -> list[str]:
        items = [item.strip() for item in value.split(",") if item.strip()]
        if not items:
            return ["all"]
        if any(item.lower() in {"all", "*"} for item in items):
            return ["all"]
        return items

    @staticmethod
    def parse_target_layers(value: str) -> int | None:
        lowered = value.strip().lower()
        if lowered in {"all", "*"}:
            return None
        count = int(lowered)
        if count < 1:
            raise ValueError("SR_TARGET_LAYERS must be a positive integer or 'all'")
        return count

    @staticmethod
    def parse_layer_indices(value: str) -> set[int] | None:
        lowered = value.strip().lower()
        if lowered in {"", "none", "null", "all", "*"}:
            return None
        indexes = {int(item.strip()) for item in value.split(",") if item.strip()}
        return indexes or None

    def target_modules_label(self) -> str:
        return ",".join(self.sr_target_modules)

    def target_layers_label(self) -> str:
        return "all" if self.target_layers is None else str(self.target_layers)

    def target_layer_indices_label(self) -> str:
        if self.target_layer_indices is None:
            return "all"
        return ",".join(str(index) for index in sorted(self.target_layer_indices))

    def build_debug_recorder(self) -> Any | None:
        if not self.bool_value(self.env.get("SR_DEBUG", self.env.get("DEBUG", "false"))):
            return None
        try:
            from importlib import import_module

            debug_module = import_module("debug.debug")
            recorder_cls = getattr(debug_module, "SrElementMetricDebugRecorder")
            return recorder_cls(self.env, method_name=self.name)
        except Exception as exc:
            try:
                from importlib import import_module

                debug_module = import_module("debug")
                recorder_cls = getattr(debug_module, "SrElementMetricDebugRecorder")
                print(f"[sr_lora] debug recorder loaded from debug.py after package import failed: {exc}", flush=True)
                return recorder_cls(self.env, method_name=self.name)
            except Exception as fallback_exc:
                print(
                    "[sr_lora] debug recorder import failed; using built-in recorder "
                    f"package_error={exc} module_error={fallback_exc}",
                    flush=True,
                )
                return globals()["SrElementMetricDebugRecorder"](self.env, method_name=self.name)


class SrElementMetricDebugRecorder:
    def __init__(self, env: Mapping[str, str], method_name: str = "sr_lora") -> None:
        self.env = dict(env)
        self.method_name = method_name
        self.enabled = self.bool_value(self.env.get("SR_DEBUG", self.env.get("DEBUG")), default=False)
        self.output_dir = Path(self.env.get("SR_DEBUG_DIR", "debug_result/sr_element_metric"))
        self.step_path = self.output_dir / "sr_element_metric_steps.jsonl"
        self.module_path = self.output_dir / "sr_element_metric_modules.jsonl"
        self.manifest_path = self.output_dir / "sr_element_metric_manifest.json"
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.write_json(
                self.manifest_path,
                {
                    "method": self.method_name,
                    "created_at": strftime("%Y%m%d_%H%M%S"),
                    "files": {
                        "steps": str(self.step_path),
                        "modules": str(self.module_path),
                    },
                    "env": {
                        "SR_DEBUG": self.env.get("SR_DEBUG"),
                        "SR_DEBUG_DIR": self.env.get("SR_DEBUG_DIR"),
                        "SR_INCLUDE_MLP": self.env.get("SR_INCLUDE_MLP"),
                        "SR_INCLUDE_ATTN": self.env.get("SR_INCLUDE_ATTN"),
                        "SR_TARGET_MODULES": self.env.get("SR_TARGET_MODULES"),
                        "SR_TARGET_LAYERS": self.env.get("SR_TARGET_LAYERS"),
                        "SR_TARGET_LAYER_INDICES": self.env.get("SR_TARGET_LAYER_INDICES"),
                        "SR_FINITE_DIFF_EPS": self.env.get("SR_FINITE_DIFF_EPS"),
                        "SR_X_CLIP": self.env.get("SR_X_CLIP"),
                        "SR_METRIC_GAIN": self.env.get("SR_METRIC_GAIN"),
                        "SR_METRIC_MIN": self.env.get("SR_METRIC_MIN"),
                        "SR_METRIC_MAX": self.env.get("SR_METRIC_MAX"),
                        "SR_METRIC_COND_MAX": self.env.get("SR_METRIC_COND_MAX"),
                        "SR_LAMBDA_LOOKAHEAD": self.env.get("SR_LAMBDA_LOOKAHEAD"),
                        "SR_LOOKAHEAD_ENABLED": self.env.get("SR_LOOKAHEAD_ENABLED"),
                        "SR_LOOKAHEAD_LR": self.env.get("SR_LOOKAHEAD_LR"),
                        "SR_LOOKAHEAD_MARGIN": self.env.get("SR_LOOKAHEAD_MARGIN"),
                        "SR_LOOKAHEAD_EVERY": self.env.get("SR_LOOKAHEAD_EVERY"),
                        "SR_LOOKAHEAD_MAX_SAMPLES": self.env.get("SR_LOOKAHEAD_MAX_SAMPLES"),
                    },
                },
            )
            print(f"[sr_lora] debug recorder active output_dir={self.output_dir}", flush=True)

    def record_step(self, *, step: int, rows: Sequence[Mapping[str, Any]]) -> None:
        if not self.enabled:
            return

        materialized = [dict(row) for row in rows]
        for row in materialized:
            self.append_jsonl(self.module_path, {"step": int(step), **row})

        active = [row for row in materialized if row.get("active")]
        summary = {
            "step": int(step),
            "method": self.method_name,
            "modules": len(materialized),
            "active_modules": len(active),
            "skipped_modules": len(materialized) - len(active),
            "skip_reasons": sorted({str(row.get("reason")) for row in materialized if not row.get("active")}),
        }
        if active:
            summary.update(
                {
                    "avg_suppression_ratio": self.mean(row["suppression_ratio"] for row in active),
                    "avg_small_denom_fraction": self.mean(row["small_denom_fraction"] for row in active),
                    "avg_metric_loss": self.mean(row["metric_loss"] for row in active),
                    "metric_min": min(row["metric_min"] for row in active),
                    "metric_max": max(row["metric_max"] for row in active),
                    "avg_x_negative_fraction": self.mean(row["x_negative_fraction"] for row in active),
                    "avg_x_positive_fraction": self.mean(row["x_positive_fraction"] for row in active),
                    "avg_lookahead_objective_gain": self.mean(
                        row.get("lookahead", {}).get("objective_gain", 0.0) for row in active
                    ),
                    "lookahead_better_fraction": self.mean(
                        1.0 if row.get("lookahead", {}).get("masked_better_than_plain") else 0.0
                        for row in active
                    ),
                }
            )
        self.append_jsonl(self.step_path, summary)

    @staticmethod
    def bool_value(value: str | None, default: bool = False) -> bool:
        if value is None:
            return default
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
        return default

    @staticmethod
    def mean(values: Any) -> float:
        items = list(values)
        return sum(float(item) for item in items) / len(items) if items else 0.0

    @staticmethod
    def write_json(path: Path, data: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def append_jsonl(path: Path, data: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")