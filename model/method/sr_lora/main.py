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
        # When explicit layer indices are provided, they are the authoritative SR target set.
        # SR_TARGET_LAYERS is only a fallback selector when SR_TARGET_LAYER_INDICES is none/all.
        self.strict_target_check = self.bool_value(self.env.get("SR_STRICT_TARGET_CHECK", "true"))
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
        self.lookahead_sync_dropout_rng = self.bool_value(
            self.env.get("SR_LOOKAHEAD_SYNC_DROPOUT_RNG", "true")
        )
        self.mask_placement = self.env.get("SR_MASK_PLACEMENT", self.env.get("SR_MASK_MODE", "backward")).strip().lower()
        if self.mask_placement not in {"backward", "forward"}:
            raise ValueError(f"SR_MASK_PLACEMENT must be backward or forward, got {self.mask_placement}")
        self._current_contrast_grads: dict[int, Any] = {}
        self._previous_boundary: dict[int, dict[str, Any]] = {}
        # Each entry is (loss_fn, exact samples used by that micro-batch, 1/grad_accum_steps).
        # Keeping the original micro-batch partition makes the virtual objective match the
        # objective that produced the accumulated contrastive gradient.
        self._lookahead_batches: list[tuple[Any, tuple[Any, ...], float]] = []
        self._previous_trainable_state: dict[str, Any] | None = None
        self.same_sample_secant = self.bool_value(self.env.get("SR_SAME_SAMPLE_SECANT", "true"))
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

        self.remember_lookahead_batch(out, grad_accum_steps)

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

    def before_optimizer_step(self, model: Any, step: int, optimizer: Any | None = None) -> dict[str, Any] | None:
        import torch

        targets = self.target_lora_b_modules(model)
        probe_batches = self.lookahead_probe_batches()
        if not targets:
            self._current_contrast_grads.clear()
            self._lookahead_batches.clear()
            return {"active": False, "reason": "no_sr_target_lora_b"}

        # The first optimizer boundary is kept as the reference model state.  From the
        # next boundary onward we estimate the secant on the *current accumulated
        # samples* at both W_t and W_{t-1}, instead of subtracting gradients from two
        # unrelated training batches.
        if self._previous_trainable_state is None:
            rows = []
            for name, module in targets:
                current_grad = self._current_contrast_grads.get(id(module))
                current_weight = module.weight.detach().float().clone()
                if current_grad is not None:
                    self._previous_boundary[id(module)] = {
                        "weight": current_weight,
                        "contrast_grad": current_grad.detach().float().clone(),
                    }
                rows.append({"active": False, "name": name, "reason": "first_boundary_skip"})
            self._previous_trainable_state = self.snapshot_trainable_state(model)
            self._current_contrast_grads.clear()
            self._lookahead_batches.clear()
            self._last_step_rows = rows
            if self.debug is not None:
                self.debug.record_step(step=step, rows=rows)
            return {"active": False, "reason": "first_boundary_skip", "modules": len(rows)}

        same_current: dict[str, Any] = {}
        same_previous: dict[str, Any] = {}
        if self.same_sample_secant and probe_batches:
            same_current, same_previous = self.same_sample_secant_gradients(
                model=model,
                targets=targets,
                probe_batches=probe_batches,
            )

        contexts: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        for name, module in targets:
            module_id = id(module)
            param_name = f"{name}.weight"
            current_grad = self._current_contrast_grads.get(module_id)
            if current_grad is None:
                rows.append({"active": False, "name": name, "reason": "no_current_grad"})
                continue

            weight = module.weight
            current_weight = weight.detach().float().clone()
            previous_weight = self._previous_trainable_state.get(param_name)
            if previous_weight is None:
                previous = self._previous_boundary.get(module_id)
                previous_weight = None if previous is None else previous.get("weight")
            if previous_weight is None:
                rows.append({"active": False, "name": name, "reason": "no_previous_weight"})
                continue
            previous_weight = previous_weight.to(device=current_weight.device, dtype=current_weight.dtype)
            delta_weight = current_weight - previous_weight

            if param_name in same_current and param_name in same_previous:
                current_secant_grad = same_current[param_name].to(device=current_weight.device, dtype=torch.float32)
                previous_secant_grad = same_previous[param_name].to(device=current_weight.device, dtype=torch.float32)
                delta_contrast = current_secant_grad - previous_secant_grad
                secant_source = "same_samples_current_vs_previous_state"
            else:
                previous = self._previous_boundary.get(module_id)
                if previous is None or previous.get("contrast_grad") is None:
                    rows.append({"active": False, "name": name, "reason": "no_previous_grad"})
                    continue
                delta_contrast = current_grad.detach().float() - previous["contrast_grad"].to(device=current_grad.device)
                secant_source = "fallback_cross_batch"

            x_raw, valid = self.secant_ratio(delta_contrast, delta_weight)
            x_clipped = self.signed_soft_clip(x_raw, self.x_clip)
            x_unit = self.l2_normalize_signed(x_clipped).detach()
            metric_diag = self.metric_for_update(module.sr_metric_raw, x_unit)
            inv_metric = metric_diag.reciprocal()

            # The real training update is modified only here.  The lookahead below uses
            # virtual parameter copies and therefore cannot mutate the live model.
            if self.mask_placement == "forward":
                self.update_forward_mask(module, inv_metric)
            elif self.mask_placement == "backward" and weight.grad is not None:
                preconditioned = (weight.grad.detach().float() * inv_metric).to(dtype=weight.grad.dtype)
                weight.grad.copy_(preconditioned)

            reg_loss, loss_parts = self.metric_regularization_loss(module, x_unit)
            reg_grad = torch.autograd.grad(reg_loss, module.sr_metric_raw, allow_unused=True)[0]
            module.sr_metric_raw.grad = None if reg_grad is None else reg_grad.detach().clone()

            contexts.append(
                {
                    "name": name,
                    "param_name": param_name,
                    "module": module,
                    "weight": weight,
                    "current_grad": current_grad,
                    "delta_weight": delta_weight,
                    "delta_contrast": delta_contrast,
                    "x_unit": x_unit,
                    "valid": valid,
                    "metric_diag": metric_diag,
                    "inv_metric": inv_metric,
                    "reg_loss": reg_loss,
                    "loss_parts": loss_parts,
                    "secant_source": secant_source,
                }
            )

        # Compare the two methods from the same W_t, with the same accumulated
        # contrastive gradient and the same exact (x, y+, y-) samples.  All SR target
        # LoRA-B parameters are changed together in the virtual branch.
        lookahead_loss, lookahead_row = self.joint_lookahead_improvement_loss(
            model=model,
            contexts=contexts,
            probe_batches=probe_batches,
            step=step,
        )
        if contexts and lookahead_row.get("active"):
            metric_params = [ctx["module"].sr_metric_raw for ctx in contexts]
            lookahead_grads = torch.autograd.grad(
                self.lambda_lookahead * lookahead_loss,
                metric_params,
                allow_unused=True,
            )
            for ctx, lookahead_grad in zip(contexts, lookahead_grads):
                if lookahead_grad is None:
                    continue
                existing = ctx["module"].sr_metric_raw.grad
                ctx["module"].sr_metric_raw.grad = (
                    lookahead_grad.detach().clone()
                    if existing is None
                    else existing + lookahead_grad.detach()
                )

        for ctx in contexts:
            current_grad = ctx["current_grad"]
            inv_metric = ctx["inv_metric"]
            metric_diag = ctx["metric_diag"]
            x_unit = ctx["x_unit"]
            valid = ctx["valid"]
            raw_grad_norm = self.norm(current_grad)
            preconditioned_norm = self.norm(current_grad * inv_metric)
            self.record_soft_mask_visualization(
                step=step,
                name=ctx["name"],
                mask=inv_metric,
                metric=metric_diag,
                x_unit=x_unit,
            )
            total_metric_loss = self.scalar(ctx["reg_loss"])
            if lookahead_row.get("active"):
                total_metric_loss += self.lambda_lookahead * self.scalar(lookahead_loss.detach())
            loss_parts = dict(ctx["loss_parts"])
            loss_parts["lookahead"] = self.scalar(lookahead_loss.detach()) if lookahead_row.get("active") else 0.0
            row = {
                "active": True,
                "name": ctx["name"],
                "mask_placement": self.mask_placement,
                "secant_source": ctx["secant_source"],
                "shape": list(ctx["weight"].shape),
                "soft_mask": self.tensor_distribution(inv_metric),
                "d_weight_norm": self.norm(ctx["delta_weight"]),
                "d_contrast_grad_norm": self.norm(ctx["delta_contrast"]),
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
                "metric_loss": total_metric_loss,
                "metric_loss_parts": loss_parts,
                "lookahead": lookahead_row,
            }
            rows.append(row)
            self._previous_boundary[id(ctx["module"])] = {
                "weight": ctx["weight"].detach().float().clone(),
                "contrast_grad": current_grad.detach().float().clone(),
            }

        # Snapshot W_t before optimizer.step(); on the next boundary it is the exact
        # previous trainable state used for same-sample secant estimation.
        self._previous_trainable_state = self.snapshot_trainable_state(model)
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
                1.0 if row.get("lookahead", {}).get("sr_better_than_contrastive") else 0.0
                for row in active_rows
            ),
            "metric_min": min(row["metric_min"] for row in active_rows),
            "metric_max": max(row["metric_max"] for row in active_rows),
            "sample_modules": active_rows[:3],
        }

    def snapshot_trainable_state(self, model: Any) -> dict[str, Any]:
        return {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad and not name.endswith("sr_metric_raw")
        }

    def same_sample_secant_gradients(
        self,
        model: Any,
        targets: Sequence[tuple[str, Any]],
        probe_batches: Sequence[tuple[Any, tuple[Any, ...], float]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        import torch

        if self._previous_trainable_state is None or not probe_batches:
            return {}, {}

        parameter_lookup = dict(model.named_parameters())
        target_param_names = [f"{name}.weight" for name, _ in targets]

        current_overrides: dict[str, Any] = {}
        current_vars = []
        for param_name in target_param_names:
            param = parameter_lookup.get(param_name)
            if param is None:
                continue
            value = param.detach().clone().requires_grad_(True)
            current_overrides[param_name] = value
            current_vars.append((param_name, value))

        previous_overrides: dict[str, Any] = {}
        previous_vars = []
        for name, value in self._previous_trainable_state.items():
            param = parameter_lookup.get(name)
            if param is None:
                continue
            restored = value.to(device=param.device, dtype=param.dtype).detach().clone()
            if name in target_param_names:
                restored.requires_grad_(True)
                previous_vars.append((name, restored))
            previous_overrides[name] = restored

        if not current_vars or not previous_vars:
            return {}, {}

        was_training = bool(getattr(model, "training", False))
        rng_state = self.capture_rng_state()
        try:
            model.eval()
            # Both branches see exactly the same samples, negatives, model mode and RNG.
            self.restore_rng_state(rng_state)
            current_objective, _ = self.lookahead_objective_batches(
                model=model,
                probe_batches=probe_batches,
                parameter_overrides=current_overrides,
                normalize=False,
            )
            current_grad_values = torch.autograd.grad(
                current_objective,
                [value for _, value in current_vars],
                allow_unused=True,
            )

            self.restore_rng_state(rng_state)
            previous_objective, _ = self.lookahead_objective_batches(
                model=model,
                probe_batches=probe_batches,
                parameter_overrides=previous_overrides,
                normalize=False,
            )
            previous_grad_values = torch.autograd.grad(
                previous_objective,
                [value for _, value in previous_vars],
                allow_unused=True,
            )
        finally:
            self.restore_rng_state(rng_state)
            model.train(was_training)

        current = {
            name: grad.detach().float()
            for (name, _), grad in zip(current_vars, current_grad_values)
            if grad is not None
        }
        previous = {
            name: grad.detach().float()
            for (name, _), grad in zip(previous_vars, previous_grad_values)
            if grad is not None
        }
        return current, previous

    def attach_sr_metric_parameters(self, model: Any) -> None:
        import torch

        targets = self.target_lora_b_modules(model)
        if not targets:
            # Still emit a persistent report before failing/returning so the resolved env is visible.
            self.verify_sr_target_selection(model, targets, metrics_attached=0)
            print("[sr_lora] no elementwise lora_B targets found; SR preconditioner disabled", flush=True)
            return

        attached = 0
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
            if self.mask_placement == "forward":
                self.ensure_forward_mask_buffer(module, shape)
                self.attach_forward_mask(module)
            metric = getattr(module, "sr_metric_raw", None)
            if metric is not None and tuple(metric.shape) == shape:
                attached += 1
            print(
                f"[sr_lora] attached elementwise metric to {name} "
                f"layer={self.layer_index(name)} shape={shape} mask_placement={self.mask_placement}",
                flush=True,
            )

        report = self.verify_sr_target_selection(model, targets, metrics_attached=attached)
        print(
            "[sr_lora] active elementwise metrics="
            f"{len(targets)} modules={self.target_modules_label()} layers={self.target_layers_label()} "
            f"layer_indices={self.target_layer_indices_label()} actual_layers={report['actual_layer_indices']} "
            f"include_mlp={self.include_mlp} include_attn={self.include_attn} "
            f"mask_placement={self.mask_placement}",
            flush=True,
        )

    def verify_sr_target_selection(
        self,
        model: Any,
        targets: Sequence[tuple[str, Any]],
        *,
        metrics_attached: int,
    ) -> dict[str, Any]:
        """Verify that env-requested SR layers are the layers that actually received SR targets.

        With explicit SR_TARGET_LAYER_INDICES, the check is strict by default: every requested
        layer must be present and no extra layer may be selected.  A JSON report is also saved
        under the experiment output directory for post-run inspection.
        """
        requested = None if self.target_layer_indices is None else set(self.target_layer_indices)
        actual = {
            index
            for name, _ in targets
            for index in [self.layer_index(name)]
            if index is not None
        }

        modules_by_layer: dict[int, list[str]] = {}
        for name, _ in targets:
            index = self.layer_index(name)
            if index is None:
                continue
            modules_by_layer.setdefault(index, []).append(name)

        module_groups_by_layer = {
            index: sorted({self.module_group_name(name) for name in names})
            for index, names in modules_by_layer.items()
        }
        missing = sorted(requested - actual) if requested is not None else []
        unexpected = sorted(actual - requested) if requested is not None else []

        report = {
            "strict": bool(self.strict_target_check),
            "resolved_env": {
                "SR_TARGET_LAYER_INDICES": self.env.get("SR_TARGET_LAYER_INDICES"),
                "SR_TARGET_LAYERS": self.env.get("SR_TARGET_LAYERS"),
                "SR_TARGET_MODULES": self.env.get("SR_TARGET_MODULES"),
                "SR_INCLUDE_MLP": self.env.get("SR_INCLUDE_MLP"),
                "SR_INCLUDE_ATTN": self.env.get("SR_INCLUDE_ATTN"),
            },
            "requested_layer_indices": None if requested is None else sorted(requested),
            "actual_layer_indices": sorted(actual),
            "requested_layer_count": None if requested is None else len(requested),
            "actual_layer_count": len(actual),
            "target_module_count": len(targets),
            "metrics_attached": int(metrics_attached),
            "missing_layer_indices": missing,
            "unexpected_layer_indices": unexpected,
            "modules_per_layer": {str(k): len(v) for k, v in sorted(modules_by_layer.items())},
            "module_groups_per_layer": {str(k): v for k, v in sorted(module_groups_by_layer.items())},
            "target_names": [name for name, _ in targets],
        }

        print(
            "[sr_lora][target-check] "
            f"requested_layers={report['requested_layer_indices']} "
            f"actual_layers={report['actual_layer_indices']} "
            f"requested_count={report['requested_layer_count']} "
            f"actual_count={report['actual_layer_count']} "
            f"target_modules={report['target_module_count']} "
            f"metrics_attached={report['metrics_attached']}",
            flush=True,
        )
        for index in sorted(modules_by_layer):
            print(
                "[sr_lora][target-check] "
                f"layer={index} modules={len(modules_by_layer[index])} "
                f"groups={module_groups_by_layer[index]}",
                flush=True,
            )

        report_path = (
            Path(self.env.get("OUTPUT_ROOT", "outputs"))
            / self.name
            / self.env.get("EVAL_DATASET", "eval")
            / "sr_target_check.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[sr_lora][target-check] report={report_path}", flush=True)

        errors = []
        if requested is not None and (missing or unexpected):
            errors.append(
                f"requested={sorted(requested)} actual={sorted(actual)} "
                f"missing={missing} unexpected={unexpected}"
            )
        if metrics_attached != len(targets):
            errors.append(f"metrics_attached={metrics_attached} target_modules={len(targets)}")
        if not targets:
            errors.append("no SR LoRA-B target modules were found")

        if errors:
            message = "; ".join(errors)
            if self.strict_target_check:
                raise RuntimeError(f"SR target verification failed: {message}. See {report_path}")
            print(f"[sr_lora][target-check] WARNING {message}", flush=True)
        else:
            print("[sr_lora][target-check] PASS", flush=True)

        return report


    def ensure_forward_mask_buffer(self, module: Any, shape: tuple[int, ...]) -> None:
        import torch

        current = getattr(module, "sr_forward_mask", None)
        if current is not None and tuple(current.shape) != tuple(shape):
            del module._buffers["sr_forward_mask"]
            current = None
        if current is None:
            module.register_buffer(
                "sr_forward_mask",
                torch.ones(shape, device=module.weight.device, dtype=torch.float32),
                persistent=False,
            )

    def attach_forward_mask(self, module: Any) -> None:
        if getattr(module, "_sr_forward_mask_wrapped", False):
            return

        import types
        import torch.nn.functional as F

        original_forward = module.forward

        def sr_forward_masked(inner_self: Any, input: Any) -> Any:
            mask = getattr(inner_self, "sr_forward_mask", None)
            if mask is None:
                return original_forward(input)
            weight = inner_self.weight
            masked_weight = weight * mask.to(device=weight.device, dtype=weight.dtype)
            bias = getattr(inner_self, "bias", None)
            return F.linear(input, masked_weight, bias)

        module.forward = types.MethodType(sr_forward_masked, module)
        module._sr_forward_mask_wrapped = True

    @staticmethod
    def update_forward_mask(module: Any, mask: Any) -> None:
        import torch

        target = mask.detach().float()
        current = getattr(module, "sr_forward_mask", None)
        if current is None or tuple(current.shape) != tuple(target.shape):
            if current is not None:
                del module._buffers["sr_forward_mask"]
            module.register_buffer(
                "sr_forward_mask",
                torch.ones_like(target, device=module.weight.device, dtype=torch.float32),
                persistent=False,
            )
            current = module.sr_forward_mask
        with torch.no_grad():
            current.copy_(target.to(device=current.device, dtype=current.dtype))

    def target_lora_b_modules(self, model: Any) -> list[tuple[str, Any]]:
        candidates = [
            (name, module)
            for name, module in model.named_modules()
            if "lora_B" in name and hasattr(module, "weight") and self.matches_target(name)
        ]
        candidates = self.filter_layer_indices(candidates)
        # Explicit SR_TARGET_LAYER_INDICES is authoritative.  Do not silently trim the
        # requested set again with SR_TARGET_LAYERS; otherwise e.g. 25,26,27 + layers=1
        # would unexpectedly keep only one layer.
        if self.target_layer_indices is not None:
            return candidates
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

        metric = 1.0 + self.metric_gain * x_unit
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

        metric = 1.0 + self.metric_gain * signal
        return torch.clamp(metric, min=self.metric_min, max=self.metric_max)

    def raw_metric_signal(self, metric_raw: Any) -> Any:
        import torch

        return torch.tanh(metric_raw.float() / max(self.metric_temp, 1e-12))

    def metric_regularization_loss(self, module: Any, x_unit: Any) -> tuple[Any, dict[str, float]]:
        import torch.nn.functional as F

        metric_raw = module.sr_metric_raw
        raw_signal = self.raw_metric_signal(metric_raw)
        learned_unit = self.l2_normalize_signed(raw_signal)
        unbounded_metric = 1.0 + self.metric_gain * learned_unit
        fit_loss = (learned_unit - x_unit).pow(2).mean()
        identity_loss = (unbounded_metric - 1.0).pow(2).mean()
        bound_loss = F.relu(self.metric_min - unbounded_metric).pow(2).mean()
        bound_loss = bound_loss + F.relu(unbounded_metric - self.metric_max).pow(2).mean()
        condition = unbounded_metric.max() / unbounded_metric.min().clamp_min(1e-12)
        condition_loss = F.relu(condition - self.condition_max).pow(2)
        loss = (
            self.lambda_fit * fit_loss
            + self.lambda_identity * identity_loss
            + self.lambda_bound * bound_loss
            + self.lambda_condition * condition_loss
        )
        return loss, {
            "fit": self.scalar(fit_loss),
            "identity": self.scalar(identity_loss),
            "bound": self.scalar(bound_loss),
            "condition": self.scalar(condition_loss),
            "condition_value": self.scalar(condition),
        }

    def remember_lookahead_batch(self, out: Any, grad_accum_steps: int) -> None:
        loss_fn = getattr(out, "loss_fn", None)
        samples = getattr(out, "samples", None)
        if loss_fn is None or samples is None:
            return
        # Keep the exact judged hallucination samples and their fixed negatives.
        self._lookahead_batches.append((loss_fn, tuple(samples), 1.0 / max(1, int(grad_accum_steps))))

    def lookahead_probe_batches(self) -> tuple[tuple[Any, tuple[Any, ...], float], ...]:
        # Do not silently take only the last micro-batch.  The accumulated gradient was
        # produced by every stored micro-batch, so both virtual branches are evaluated
        # on the same complete set.
        return tuple(self._lookahead_batches)

    def joint_lookahead_improvement_loss(
        self,
        model: Any,
        contexts: Sequence[Mapping[str, Any]],
        probe_batches: Sequence[tuple[Any, tuple[Any, ...], float]],
        step: int,
    ) -> tuple[Any, dict[str, Any]]:
        import torch
        import torch.nn.functional as F

        metric_params = [ctx["module"].sr_metric_raw for ctx in contexts]
        zero = metric_params[0].new_tensor(0.0) if metric_params else torch.tensor(0.0)
        if not self.lookahead_enabled:
            return zero, {"active": False, "reason": "disabled"}
        if step % self.lookahead_every != 0:
            return zero, {"active": False, "reason": "interval"}
        if not probe_batches:
            return zero, {"active": False, "reason": "no_probe_batches"}

        contrastive_overrides: dict[str, Any] = {}
        sr_overrides: dict[str, Any] = {}
        used_targets = 0
        for ctx in contexts:
            name = str(ctx["name"])
            module = ctx["module"]
            contrast_grad = ctx["current_grad"]
            param_name = str(ctx["param_name"])
            weight = module.weight
            update = contrast_grad.detach().to(device=weight.device, dtype=torch.float32)

            # Forward value == the exact mask that is applied to the real gradient.
            # Backward derivative == the learned metric path, so lookahead can still
            # train sr_metric_raw.  This avoids evaluating an identity mask when the
            # live update is currently using the x-derived fallback mask.
            actual_inv_metric = ctx["inv_metric"].detach().to(device=weight.device, dtype=torch.float32)
            learned_inv_metric = self.metric_from_raw(module.sr_metric_raw).reciprocal()
            candidate_inv_metric = actual_inv_metric + (learned_inv_metric - learned_inv_metric.detach())
            masked_update = update * candidate_inv_metric

            contrastive_overrides[param_name] = (
                weight - self.lookahead_lr * update.to(device=weight.device, dtype=weight.dtype)
            ).detach()
            sr_overrides[param_name] = weight - self.lookahead_lr * masked_update.to(
                device=weight.device,
                dtype=weight.dtype,
            )

            # For the forward-mask ablation, compare an unmasked contrastive branch to
            # the exact candidate SR mask, again from the same W_t.
            if self.mask_placement == "forward":
                buffer_name = f"{name}.sr_forward_mask"
                contrastive_overrides[buffer_name] = torch.ones_like(actual_inv_metric, dtype=torch.float32)
                sr_overrides[buffer_name] = candidate_inv_metric
            used_targets += 1

        if used_targets == 0:
            return zero, {"active": False, "reason": "no_target_grad"}

        probe_samples = sum(len(samples) for _, samples, _ in probe_batches)
        was_training = bool(getattr(model, "training", False))
        rng_state = self.capture_rng_state()
        try:
            # Lookahead is a deterministic diagnostic/metric-learning comparison.
            # train-time dropout must not make the two virtual branches differ.
            model.eval()
            self.restore_rng_state(rng_state)
            with torch.no_grad():
                contrastive_objective, contrastive_parts = self.lookahead_objective_batches(
                    model=model,
                    probe_batches=probe_batches,
                    parameter_overrides=contrastive_overrides,
                    normalize=True,
                )

            self.restore_rng_state(rng_state)
            sr_objective, sr_parts = self.lookahead_objective_batches(
                model=model,
                probe_batches=probe_batches,
                parameter_overrides=sr_overrides,
                normalize=True,
            )
        finally:
            self.restore_rng_state(rng_state)
            model.train(was_training)

        objective_delta = sr_objective - contrastive_objective.detach()
        loss = F.softplus(objective_delta + self.lookahead_margin)
        gain = contrastive_objective.detach() - sr_objective.detach()
        better = bool(self.scalar(gain) > 0.0)
        row = {
            "active": True,
            "comparison": "contrastive_vs_sr_same_samples_same_start_joint_targets",
            "probe_samples": probe_samples,
            "probe_microbatches": len(probe_batches),
            "targets": used_targets,
            "contrastive_gold_ce": self.float_or_none(contrastive_parts["gold_ce"]),
            "sr_gold_ce": self.float_or_none(sr_parts["gold_ce"]),
            "contrastive_hall_ce": self.float_or_none(contrastive_parts["hall_ce"]),
            "sr_hall_ce": self.float_or_none(sr_parts["hall_ce"]),
            "contrastive_objective": self.scalar(contrastive_objective),
            "sr_objective": self.scalar(sr_objective.detach()),
            "objective_gain": self.scalar(gain),
            "sr_better_than_contrastive": better,
            "loss": self.scalar(loss.detach()),
            "margin": self.lookahead_margin,
            # Backward-compatible aliases for existing debug scripts.
            "plain_gold_ce": self.float_or_none(contrastive_parts["gold_ce"]),
            "masked_gold_ce": self.float_or_none(sr_parts["gold_ce"]),
            "plain_hall_ce": self.float_or_none(contrastive_parts["hall_ce"]),
            "masked_hall_ce": self.float_or_none(sr_parts["hall_ce"]),
            "plain_objective": self.scalar(contrastive_objective),
            "masked_objective": self.scalar(sr_objective.detach()),
            "masked_better_than_plain": better,
        }
        return loss, row

    @staticmethod
    def _weighted_mean_or_none(total: Any | None, total_weight: float) -> Any | None:
        if total is None:
            return None
        return total / max(total_weight, 1e-12)

    def lookahead_objective_batches(
        self,
        model: Any,
        probe_batches: Sequence[tuple[Any, tuple[Any, ...], float]],
        parameter_overrides: Mapping[str, Any],
        normalize: bool,
    ) -> tuple[Any, dict[str, Any]]:
        objective_total = None
        gold_total = None
        hall_total = None
        weight_total = 0.0

        for loss_fn, samples, scale in probe_batches:
            if not samples:
                continue
            gold_ce = self.lookahead_sequence_ce(
                model=model,
                loss_fn=loss_fn,
                samples=samples,
                answer_attr="answer",
                parameter_overrides=parameter_overrides,
                chunk_size=self.lookahead_chunk_size,
            )
            hall_samples = tuple(sample for sample in samples if getattr(sample, "hallucinated_answer", ""))
            hall_ce = None
            batch_objective = gold_ce
            if hall_samples:
                hall_ce = self.lookahead_sequence_ce(
                    model=model,
                    loss_fn=loss_fn,
                    samples=hall_samples,
                    answer_attr="hallucinated_answer",
                    parameter_overrides=parameter_overrides,
                    chunk_size=self.lookahead_chunk_size,
                )
                batch_objective = batch_objective - hall_ce

            scaled_objective = batch_objective * scale
            scaled_gold = gold_ce * scale
            objective_total = scaled_objective if objective_total is None else objective_total + scaled_objective
            gold_total = scaled_gold if gold_total is None else gold_total + scaled_gold
            if hall_ce is not None:
                scaled_hall = hall_ce * scale
                hall_total = scaled_hall if hall_total is None else hall_total + scaled_hall
            weight_total += scale

        if objective_total is None or gold_total is None:
            # This path is only reachable with empty probe batches.
            first = next(iter(parameter_overrides.values()))
            zero = first.new_tensor(0.0)
            return zero, {"gold_ce": zero, "hall_ce": None}

        if normalize:
            objective_total = objective_total / max(weight_total, 1e-12)
            gold_total = gold_total / max(weight_total, 1e-12)
            if hall_total is not None:
                hall_total = hall_total / max(weight_total, 1e-12)
        return objective_total, {"gold_ce": gold_total, "hall_ce": hall_total}

    @staticmethod
    def lookahead_sequence_ce(
        model: Any,
        loss_fn: Any,
        samples: Sequence[Any],
        answer_attr: str,
        parameter_overrides: Mapping[str, Any],
        chunk_size: int = 2,
    ) -> Any:
        import torch
        import torch.nn.functional as F
        from torch.func import functional_call

        reference = next(iter(parameter_overrides.values()))
        total_loss = reference.new_tensor(0.0, dtype=torch.float32)
        total_tokens = reference.new_tensor(0.0, dtype=torch.float32)
        chunk_size = max(1, int(chunk_size))
        for start in range(0, len(samples), chunk_size):
            chunk = samples[start : start + chunk_size]
            if not chunk:
                continue
            batch = loss_fn.build_lm_batch(chunk, answer_attr)
            batch = {key: value.to(reference.device) for key, value in batch.items()}
            logits = functional_call(
                model,
                dict(parameter_overrides),
                (),
                {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
                strict=False,
            ).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            total_loss = total_loss + F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            ).float()
            total_tokens = total_tokens + shift_labels.ne(-100).sum().to(
                device=total_tokens.device,
                dtype=total_tokens.dtype,
            )
        return total_loss / total_tokens.clamp_min(1.0)

    @staticmethod
    def capture_rng_state() -> dict[str, Any]:
        import torch

        state: dict[str, Any] = {
            "cpu": torch.get_rng_state().clone(),
            "cuda": None,
        }
        if torch.cuda.is_available():
            state["cuda"] = [rng.clone() for rng in torch.cuda.get_rng_state_all()]
        return state

    @staticmethod
    def restore_rng_state(state: Mapping[str, Any]) -> None:
        import torch

        torch.set_rng_state(state["cpu"])
        cuda_states = state.get("cuda")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)

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
                    "SR_LOOKAHEAD_SYNC_DROPOUT_RNG": self.env.get("SR_LOOKAHEAD_SYNC_DROPOUT_RNG", "true"),
                    "SR_SAME_SAMPLE_SECANT": self.env.get("SR_SAME_SAMPLE_SECANT", "true"),
                    "SR_MASK_PLACEMENT": self.env.get("SR_MASK_PLACEMENT"),
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
                        "SR_LOOKAHEAD_SYNC_DROPOUT_RNG": self.env.get("SR_LOOKAHEAD_SYNC_DROPOUT_RNG", "true"),
                    "SR_SAME_SAMPLE_SECANT": self.env.get("SR_SAME_SAMPLE_SECANT", "true"),
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