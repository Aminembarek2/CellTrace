"""Fixed model profiles chosen in advance. Runtime roles cannot change them."""

from dataclasses import dataclass

PRIMARY_MODEL_ID = "deepseek/deepseek-v4.1-flash"
COMPARISON_MODEL_IDS = ("z-ai/glm-5.3-flash", "google/gemini-3.8-flash")


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    provider_slug: str
    provider_name: str

    def routing(self) -> dict[str, object]:
        return {
            "only": [self.provider_slug],
            "order": [self.provider_slug],
            "allow_fallbacks": False,
            "require_parameters": True,
        }


MODEL_PROFILES = {
    PRIMARY_MODEL_ID: ModelProfile(PRIMARY_MODEL_ID, "deepseek", "DeepSeek"),
    COMPARISON_MODEL_IDS[0]: ModelProfile(COMPARISON_MODEL_IDS[0], "deepinfra", "DeepInfra"),
    COMPARISON_MODEL_IDS[1]: ModelProfile(
        COMPARISON_MODEL_IDS[1], "google-ai-studio", "Google AI Studio"
    ),
}


def model_profile(model_id: str) -> ModelProfile:
    try:
        return MODEL_PROFILES[model_id]
    except KeyError:
        raise ValueError(f"model_id must be one of {list(MODEL_PROFILES)}") from None
