# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import os
import traceback
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import ray
import torch
import torchvision.transforms as T
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import import_external_libs
from verl.utils.net_utils import get_free_port
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.utils import run_uvicorn
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from vllm.entrypoints.openai.api_server import build_app
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.inputs.data import OmniCustomPrompt, OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.outputs import OmniRequestOutput

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.utils.vllm_omni import VLLMOmniHijack
from verl_omni.workers.config import DiffusionModelConfig, DiffusionRolloutConfig
from verl_omni.workers.rollout.replica import DiffusionOutput

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class vLLMOmniHttpServer(vLLMHttpServer):
    """vLLM-Omni http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    # -----------------------------------------------------------------------
    # Initialisation hooks
    # -----------------------------------------------------------------------

    def _init_model_config(self, model_config):
        """Use DiffusionModelConfig instead of HFModelConfig."""
        return omega_conf_to_dataclass(model_config, dataclass_type=DiffusionModelConfig)

    def _validate_configs(self) -> None:
        """No-op: diffusion models don't have max_position_embeddings."""
        pass

    def _post_init(self, cuda_visible_devices: str) -> None:
        """Omni-specific post-init: create PIL→tensor converter, then log."""
        self._to_tensor = T.PILToTensor()
        super()._post_init(cuda_visible_devices)

    # -----------------------------------------------------------------------
    # launch_server hooks
    # -----------------------------------------------------------------------

    def _get_override_generation_config(self) -> dict:
        """Diffusion models have no LLM sampling params; return empty dict."""
        return {}

    def _get_engine_kwargs_key(self) -> str:
        return "vllm_omni"

    def _get_worker_extension_cls(self) -> str:
        return "verl_omni.workers.rollout.vllm_rollout.utils.vLLMOmniColocateWorkerExtension"

    def _get_cli_modules(self) -> list:
        import vllm_omni.entrypoints.cli.serve
        return [vllm_omni.entrypoints.cli.serve]

    def _get_cli_description(self) -> str:
        return "vLLM-Omni CLI"

    # TODO: drop it after updating verl pin (at least 5ff595ac9fcb4)
    async def launch_server(self, master_address: str = None, master_port: int = None, dp_rpc_port: int = None):
        """Launch vLLM-Omni engine; coerce null ``rollout.seed`` for engine init only.

        Upstream verl uses ``config.get("seed", 0)``, but Hydra ``seed: null`` sets the
        attribute to None, so the default is not applied and launch crashes with
        ``replica_rank + None``. Training rollout seeding stays unset via meta_info.
        """
        import sys
        print(f"[DEBUG] vLLMOmniHttpServer.launch_server called, replica_rank={self.replica_rank}, node_rank={self.node_rank}", flush=True)
        print(f"[DEBUG] vLLMOmniHttpServer.launch_server called", file=sys.stderr, flush=True)
        original_get = self.config.get

        def get_with_engine_seed_default(key: str, default: Any = None) -> Any:
            if key == "seed":
                value = original_get(key, default)
                return 0 if value is None else value
            return original_get(key, default)

        self.config.get = get_with_engine_seed_default
        try:
            await super().launch_server(master_address, master_port, dp_rpc_port)
            print(f"[DEBUG] vLLMOmniHttpServer.launch_server completed successfully, replica_rank={self.replica_rank}", flush=True)
        except Exception as e:
            print(f"[DEBUG] vLLMOmniHttpServer.launch_server FAILED with exception: {e}", flush=True)
            traceback.print_exc()
            raise
        finally:
            # BaseConfig is frozen; pop the shadowed get instead of reassigning it.
            self.config.__dict__.pop("get", None)

    # -----------------------------------------------------------------------
    # Server lifecycle
    # -----------------------------------------------------------------------

    async def run_server(self, args: argparse.Namespace):
        import sys
        print(f"[DEBUG] vLLMOmniHttpServer.run_server START, replica_rank={self.replica_rank}", flush=True)
        try:
            engine_args = OmniEngineArgs.from_cli_args(args)
            engine_args = asdict(engine_args)
            print(f"[DEBUG] run_server: engine_args parsed", flush=True)

            import_external_libs(self.config.external_lib)
            pipeline_path = VllmOmniPipelineBase.get_pipeline_path(
                architecture=self.model_config.architecture,
                algorithm=self.model_config.algorithm,
            )
            print(f"[DEBUG] run_server: pipeline_path={pipeline_path}", flush=True)
            # TODO (mike): read custom_pipeline from engine_args
            if pipeline_path is not None:
                engine_args["enable_dummy_pipeline"] = True
                engine_args["custom_pipeline_args"] = {"pipeline_class": pipeline_path}

            diffusion_master_port, diffusion_master_sock = get_free_port("127.0.0.1", with_alive_sock=True)
            diffusion_master_sock.close()

            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(diffusion_master_port)
            logger.info("Using MASTER_PORT=%s for vLLM-Omni diffusion workers", os.environ["MASTER_PORT"])

            # Apply before AsyncOmni builds OmniDiffusionConfig in this process.
            VLLMOmniHijack.hijack()
            # For RL training with a custom pipeline, use the single-stage
            # deploy config so we skip the heavyweight LLM stage init.
            if engine_args.get("enable_dummy_pipeline"):
                import json as _json
                model_path = engine_args.get("model", "")
                config_json = os.path.join(model_path, "config.json")
                if os.path.isfile(config_json):
                    with open(config_json) as _f:
                        _model_type = _json.load(_f).get("model_type", "")
                    import vllm_omni as _vo
                    _ss = os.path.join(os.path.dirname(_vo.__file__), "deploy", f"{_model_type}_single_stage.yaml")
                    if os.path.isfile(_ss):
                        engine_args["deploy_config"] = _ss
                        print(f"[DEBUG] run_server: using single-stage deploy config: {_ss}", flush=True)
            engine_args["init_timeout"] = 1800
            engine_args["stage_init_timeout"] = 900
            print(f"[DEBUG] run_server: init_timeout={engine_args.get('init_timeout')}, "
                  f"stage_init_timeout={engine_args.get('stage_init_timeout')}, "
                  f"deploy_config={engine_args.get('deploy_config', 'NONE')}, "
                  f"model={engine_args.get('model', 'NONE')}", flush=True)
            print(f"[DEBUG] run_server: about to create AsyncOmni engine", flush=True)
            engine_client = AsyncOmni(**engine_args)
            print(f"[DEBUG] run_server: AsyncOmni engine created", flush=True)
            app = build_app(args)
            print(f"[DEBUG] run_server: app built, about to init app state", flush=True)
            await omni_init_app_state(engine_client, app.state, args)
            print(f"[DEBUG] run_server: app state initialized", flush=True)

            self.engine = engine_client
            self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)
            print(f"[DEBUG] run_server: uvicorn running on port {self._server_port}", flush=True)
        except Exception as e:
            print(f"[DEBUG] vLLMOmniHttpServer.run_server FAILED: {e}", flush=True)
            print(f"[DEBUG] Full traceback:", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            traceback.print_exc()
            raise

    async def run_headless(self, args: argparse.Namespace):
        """Run headless server in a separate thread."""
        # TODO (mike): support multi node
        raise NotImplementedError("vLLM-Omni headless mode is not implemented yet.")

    # -----------------------------------------------------------------------
    # wake_up hook: Omni does not restore KV cache on wake-up
    # -----------------------------------------------------------------------

    def _get_wake_up_tags(self) -> list[str]:
        return ["weights"]

    async def wake_up(self, tags: list[str] | None = None):
        """Override parent to use collective_rpc instead of engine.wake_up().

        The parent (verl ``1927ad33``+) calls ``self.engine.wake_up(tags=...)``
        which triggers CUDA initialisation in this HTTP server process when
        running under vLLM-Omni (AsyncOmni engine).
        Use ``collective_rpc`` instead.

        # TODO (long): drop this override once vllm-omni wake_up
        without triggering GPU initialisation.
        """
        if self.node_rank != 0:
            return
        await self.engine.collective_rpc(
            "wake_up", kwargs={"tags": tags if tags is not None else self._get_wake_up_tags()}
        )

    async def _sleep_hybrid(self):
        """Preserve non-actor pipeline weights during hybrid training sleep.

        vLLM-Omni diffusion pipelines include components such as the text
        encoder and VAE that are loaded by the rollout server, but are not part
        of the trainable actor and therefore are not included in full-model
        weight syncs. Use level-1 sleep so those weights are offloaded and can
        be restored on wake-up instead of discarded by level-2 sleep.
        """
        # TODO (andy): use `sleep_level=2` in the future when the
        #  trainer side incorporates the whole components of the model.
        await self.engine.collective_rpc("sleep", kwargs={"level": 1})
        await self.engine.reset_encoder_cache()

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        negative_prompt_ids: Optional[list[int]] = None,
        priority: int = 0,
    ) -> DiffusionOutput:
        """Generate sequence with token-in-image-out."""
        prompt_ids = normalize_token_ids(prompt_ids)

        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data

        # Add lora request
        lora_request = None
        if self.lora_as_adapter:
            # Make sure we also check that the lora is already loaded in the engine
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        # Build OmniCustomPrompt with pre-tokenized IDs
        custom_prompt: OmniCustomPrompt = {"prompt_ids": prompt_ids}
        if negative_prompt_ids is not None:
            custom_prompt["negative_prompt_ids"] = negative_prompt_ids
        if multi_modal_data:
            custom_prompt["extra_args"] = {"multi_modal_data": multi_modal_data}

        # Build OmniDiffusionSamplingParams from the incoming dict
        sampling_kwargs: dict[str, Any] = {}
        extra_args: dict[str, Any] = {}
        for k, v in sampling_params.items():
            if hasattr(OmniDiffusionSamplingParams, k):
                sampling_kwargs[k] = v
            else:
                extra_args[k] = v
        sampling_kwargs["extra_args"] = extra_args
        if lora_request is not None:
            sampling_kwargs["lora_request"] = lora_request
        diffusion_sampling_params = OmniDiffusionSamplingParams(**sampling_kwargs)

        # Call AsyncOmni.generate() with the correct API
        generator = self.engine.generate(
            prompt=custom_prompt,
            request_id=request_id,
            sampling_params_list=[diffusion_sampling_params],
        )

        # Get final response
        final_res: Optional[OmniRequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None
        diffusion_output = final_res.images[0]
        if isinstance(diffusion_output, torch.Tensor):
            diffusion_output = diffusion_output.float()
        elif isinstance(diffusion_output, np.ndarray):
            diffusion_output = torch.from_numpy(diffusion_output).float()
        else:
            diffusion_output = self._to_tensor(diffusion_output).float() / 255.0

        # Extract extra data from custom_output (populated by DiffusionEngine)
        mm_output = final_res.custom_output or {}

        if sampling_params.get("logprobs", False):
            all_log_probs = mm_output.get("all_log_probs")
            log_probs = all_log_probs[0] if all_log_probs is not None else None
        else:
            log_probs = None

        all_latents = mm_output.get("all_latents")
        all_timesteps = mm_output.get("all_timesteps")
        prompt_embeds = mm_output.get("prompt_embeds")
        prompt_embeds_mask = mm_output.get("prompt_embeds_mask")
        negative_prompt_embeds = mm_output.get("negative_prompt_embeds")
        negative_prompt_embeds_mask = mm_output.get("negative_prompt_embeds_mask")
        latents_clean = mm_output.get("latents_clean")
        train_timesteps = mm_output.get("train_timesteps")
        # Bagel i2i condition embeddings (optional)
        cond_vae_embeds = mm_output.get("cond_vae_embeds")
        cond_vit_embeds = mm_output.get("cond_vit_embeds")

        # TODO(andy): refactor later.
        extra_fields = {
            "all_latents": all_latents[0] if all_latents is not None else None,
            "all_timesteps": all_timesteps[0] if all_timesteps is not None else None,
            "latents_clean": latents_clean[0] if latents_clean is not None else None,
            "train_timesteps": train_timesteps[0] if train_timesteps is not None else None,
            "prompt_embeds": prompt_embeds[0] if prompt_embeds is not None else None,
            "prompt_embeds_mask": prompt_embeds_mask[0] if prompt_embeds_mask is not None else None,
            "negative_prompt_embeds": negative_prompt_embeds[0] if negative_prompt_embeds is not None else None,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask[0]
            if negative_prompt_embeds_mask is not None
            else None,
            "cond_vae_embeds": cond_vae_embeds[0] if cond_vae_embeds is not None else None,
            "cond_vit_embeds": cond_vit_embeds[0] if cond_vit_embeds is not None else None,
            "global_steps": self.global_steps,
        }

        # Determine stop reason from finish_reason
        if final_res.request_output is not None and hasattr(final_res.request_output, "finish_reason"):
            finish_reason = final_res.request_output.finish_reason or "stop"
        else:
            finish_reason = "stop"

        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason  # for more stop reason in the future

        num_preempted = None
        if final_res.request_output is not None and hasattr(final_res.request_output, "num_preempted"):
            num_preempted = final_res.request_output.num_preempted

        return DiffusionOutput(
            diffusion_output=diffusion_output,
            log_probs=log_probs,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )

    async def wait_for_requests_to_drain(self):
        # TODO (mike): implement this once DP is supported.
        pass


class vLLMOmniReplica(vLLMReplica):
    def __init__(
        self,
        replica_rank: int,
        config: DiffusionRolloutConfig,
        model_config: DiffusionModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(vLLMOmniHttpServer)

    def _validate_launch_requirements(self) -> None:
        """No-op: the parent check validates vllm.__version__ which is
        irrelevant for vllm-omni (a separate package)."""
        pass

    def _get_server_name_prefix(self) -> str:
        return "vllm_omni_"
