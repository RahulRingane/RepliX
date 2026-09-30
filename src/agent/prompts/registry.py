from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Prompt:
    name: str
    model: str
    version: str
    content: str


def _get_module(name: str, model: str):
    model_file = model.replace("-", "_")

    return import_module(f"src.agent.prompts.{name}.{model_file}")


def get_prompt(name: str, model: str, version: str) -> Prompt:
    module = _get_module(name, model)

    prompt_constant = f"{name.upper()}_PROMPT_{version.upper()}"

    try:
        content = getattr(module, prompt_constant)
    except AttributeError:
        raise KeyError(f"No prompt version '{version}' for {name}/{model}")

    return Prompt(
        name=name,
        model=model,
        version=version,
        content=content,
    )


def get_active_prompt(name: str, model: str) -> Prompt:
    module = _get_module(name, model)

    try:
        version = module.ACTIVE_VERSION
    except AttributeError:
        raise KeyError(f"No active version defined for {name}/{model}")

    return get_prompt(name, model, version)
