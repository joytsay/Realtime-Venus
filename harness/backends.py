"""Provider selection for three independent model roles; General supports Codex and llama.cpp."""
from harness.config import ModelCallConfig, Settings
from harness.llm.backends import CodexPlannerBackend
from harness.llm.delegate import CodexDirectAndPolish, ConfiguredDirectAndPolish
from harness.llm.planner import PlannerClient


def codex_settings(setup, call):
    return Settings(
        codex_path=setup.general.command[0],
        codex_model=call.model or setup.general.model or "",
        codex_workspace=setup.general.workspace,
        codex_timeout_seconds=call.timeout_s,
        codex_effort=call.effort,
    )


def model_backend(setup, call):
    if call.provider == "llamacpp":
        from harness.llm.llamacpp import LlamaCppBackend
        return LlamaCppBackend(setup.llamacpp, model=call.model, timeout_s=call.timeout_s)
    if call.provider == "gemini":
        from harness.llm.gemini import GeminiBackend
        return GeminiBackend(api_key=setup.gemini.resolved_key(),
                             model=call.model or setup.gemini.model, timeout_s=call.timeout_s)
    return CodexPlannerBackend(codex_settings(setup, call))


def task_backends(setup):
    polish = model_backend(setup, setup.responses)
    multimodal = model_backend(setup, setup.multimodal)
    if setup.multimodal.provider == "llamacpp":
        from harness.llm.llamacpp import LlamaCppDirect
        direct = LlamaCppDirect(multimodal)
    elif setup.multimodal.provider == "gemini":
        from harness.llm.gemini import GeminiDirect
        direct = GeminiDirect(multimodal)
    else:
        direct = CodexDirectAndPolish(multimodal)
    # A dormant Gemini router must not require a key or create requests in General mode.
    router = model_backend(setup, setup.routing) if setup.routing.mode == "auto" else polish
    # Retain the existing Codex skill-planning path; changing Polish must not switch it.
    skill_call = setup.responses if setup.responses.provider == "codex" else ModelCallConfig()
    skill = (model_backend(setup, ModelCallConfig(provider="llamacpp"))
             if setup.general_provider == "llamacpp"
             else CodexPlannerBackend(codex_settings(setup, skill_call)))
    planner = PlannerClient(codex_settings(setup, ModelCallConfig()),
                            backend=skill, router_backend=router, routing_mode=setup.routing.mode)
    return ConfiguredDirectAndPolish(direct, CodexDirectAndPolish(polish)), planner
