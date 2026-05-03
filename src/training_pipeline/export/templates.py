"""Chat templates for rendering trajectories as prompt strings.

The export layer always emits a structured ``messages`` list — that's what
``SFTTrainer`` / ``DPOTrainer`` already accept directly. These Jinja-rendered
strings are useful for:

- generating ``text`` columns for trainers that need a single string
- writing the prompt half of a DPO record when the underlying tokenizer
  doesn't ship a chat template
- previewing what the model will actually see at training time
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from jinja2 import Environment, StrictUndefined

from training_pipeline.schemas.exports import SFTMessage

# Each template is a Jinja string that renders ``messages``. Keep them tight;
# trainers care about exact byte sequences.

CHATML = """{%- for m in messages -%}
<|im_start|>{{ m.role }}
{%- if m.role == 'assistant' and m.tool_calls -%}
{%- if m.content %}
{{ m.content }}{% endif %}
{%- for tc in m.tool_calls %}
<tool_call>{"name": "{{ tc.name }}", "arguments": {{ tc.arguments | tojson }}}</tool_call>
{%- endfor %}
{%- elif m.role == 'tool' %}
{{ m.content }}
{%- else %}
{{ m.content or '' }}
{%- endif %}<|im_end|>
{% endfor -%}
{%- if add_generation_prompt -%}<|im_start|>assistant
{%- endif -%}"""


LLAMA3 = """{%- for m in messages -%}
<|start_header_id|>{{ m.role }}<|end_header_id|>

{%- if m.role == 'assistant' and m.tool_calls %}
{%- if m.content %}
{{ m.content }}
{%- endif %}
{%- for tc in m.tool_calls %}
<|python_tag|>{"name": "{{ tc.name }}", "parameters": {{ tc.arguments | tojson }}}<|eom_id|>
{%- endfor -%}
{%- elif m.role == 'tool' %}
{{ m.content }}
{%- else %}
{{ m.content or '' }}
{%- endif %}<|eot_id|>
{% endfor -%}
{%- if add_generation_prompt -%}<|start_header_id|>assistant<|end_header_id|>

{%- endif -%}"""


# Plain (no special tokens) — useful when training a base model with no chat scaffolding.
PLAIN = """{%- for m in messages -%}
{{ m.role | upper }}: {%- if m.role == 'assistant' and m.tool_calls -%}
{%- if m.content %} {{ m.content }}{% endif %}
{%- for tc in m.tool_calls %}
TOOL_CALL({{ tc.name }}, {{ tc.arguments | tojson }}){%- endfor -%}
{%- elif m.role == 'tool' %} TOOL_RESULT({{ m.content }})
{%- else %} {{ m.content or '' }}
{%- endif %}
{% endfor -%}
{%- if add_generation_prompt -%}ASSISTANT:{%- endif -%}"""


KNOWN_TEMPLATES: dict[str, str] = {
    "chatml": CHATML,
    "llama3": LLAMA3,
    "plain": PLAIN,
}


def template_for(name: str) -> str:
    if name not in KNOWN_TEMPLATES:
        raise KeyError(
            f"Unknown chat template {name!r}. Known: {sorted(KNOWN_TEMPLATES)}"
        )
    return KNOWN_TEMPLATES[name]


def apply_template(
    messages: Iterable[SFTMessage | dict[str, Any]],
    *,
    template: str = "chatml",
    add_generation_prompt: bool = False,
) -> str:
    """Render messages with the named template (or a raw template string)."""
    template_src = (
        KNOWN_TEMPLATES.get(template, template) if template in KNOWN_TEMPLATES else template
    )
    env = Environment(undefined=StrictUndefined, autoescape=False)
    env.filters.setdefault("tojson", lambda v, **kw: json.dumps(v, default=str))
    tpl = env.from_string(template_src)
    msg_list = [
        m.model_dump(exclude_none=True) if isinstance(m, SFTMessage) else m
        for m in messages
    ]
    return tpl.render(messages=msg_list, add_generation_prompt=add_generation_prompt)
