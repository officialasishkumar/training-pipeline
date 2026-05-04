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
from collections.abc import Iterable
from typing import Any

from jinja2 import Environment

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


# Qwen2 / Qwen2.5 use ChatML-style tokens with a separate ``<tool_call>`` JSON
# envelope and ``<tool_response>`` for results. Most Qwen tokenizers ship a
# chat_template embedded — this string mirrors that shape so previews match.
QWEN = """{%- for m in messages -%}
<|im_start|>{{ m.role }}
{%- if m.role == 'assistant' and m.tool_calls -%}
{%- if m.content %}
{{ m.content }}{% endif %}
{%- for tc in m.tool_calls %}
<tool_call>
{"name": "{{ tc.name }}", "arguments": {{ tc.arguments | tojson }}}
</tool_call>
{%- endfor %}
{%- elif m.role == 'tool' %}
<tool_response>
{{ m.content }}
</tool_response>
{%- else %}
{{ m.content or '' }}
{%- endif %}<|im_end|>
{% endfor -%}
{%- if add_generation_prompt -%}<|im_start|>assistant
{%- endif -%}"""


# Gemma instruct prepends a BOS-like ``<start_of_turn>`` per role and ends with
# ``<end_of_turn>``. Gemma has no first-class tool-call slot, so we serialise
# tool calls/results into the message body — this matches what the tokenizer's
# default chat_template does for tool-using fine-tunes.
GEMMA = """{%- for m in messages -%}
<start_of_turn>{{ 'model' if m.role == 'assistant' else m.role }}
{%- if m.role == 'assistant' and m.tool_calls -%}
{%- if m.content %}
{{ m.content }}{% endif %}
{%- for tc in m.tool_calls %}
```tool_code
{"name": "{{ tc.name }}", "args": {{ tc.arguments | tojson }}}
```
{%- endfor %}
{%- elif m.role == 'tool' %}
```tool_result
{{ m.content }}
```
{%- else %}
{{ m.content or '' }}
{%- endif %}<end_of_turn>
{% endfor -%}
{%- if add_generation_prompt -%}<start_of_turn>model
{%- endif -%}"""


# Mistral / Mixtral instruct uses ``[INST] ... [/INST]`` framing. Tool calling
# follows Mistral's ``[TOOL_CALLS]`` / ``[TOOL_RESULTS]`` token convention; tool
# arguments are emitted as a JSON list to match transformers' Mistral template.
MISTRAL = """{%- for m in messages -%}
{%- if m.role == 'system' -%}
[INST] {{ m.content }} [/INST]
{%- elif m.role == 'user' -%}
[INST] {{ m.content }} [/INST]
{%- elif m.role == 'assistant' -%}
{%- if m.tool_calls -%}
{%- if m.content %} {{ m.content }}{% endif %}
[TOOL_CALLS] [{%- for tc in m.tool_calls %}{"name": "{{ tc.name }}", "arguments": {{ tc.arguments | tojson }}}{%- if not loop.last -%}, {%- endif -%}{%- endfor %}]</s>
{%- else -%}
 {{ m.content or '' }}</s>
{%- endif %}
{%- elif m.role == 'tool' -%}
[TOOL_RESULTS] {"content": {{ m.content | tojson }}, "call_id": "{{ m.tool_call_id }}"} [/TOOL_RESULTS]
{%- endif %}
{% endfor -%}"""


KNOWN_TEMPLATES: dict[str, str] = {
    "chatml": CHATML,
    "llama3": LLAMA3,
    "plain": PLAIN,
    "qwen": QWEN,
    "gemma": GEMMA,
    "mistral": MISTRAL,
}


def template_for(name: str) -> str:
    if name not in KNOWN_TEMPLATES:
        raise KeyError(f"Unknown chat template {name!r}. Known: {sorted(KNOWN_TEMPLATES)}")
    return KNOWN_TEMPLATES[name]


def apply_template(
    messages: Iterable[SFTMessage | dict[str, Any]],
    *,
    template: str = "chatml",
    add_generation_prompt: bool = False,
) -> str:
    """Render messages with the named template (or a raw template string).

    Renders with ``None`` fields preserved so templates can branch on them
    (``{% if m.tool_calls %}``). Strict-undefined is intentionally off here:
    real-world chat templates often access optional fields and tolerating
    missing keys keeps custom templates usable.
    """
    template_src = (
        KNOWN_TEMPLATES.get(template, template) if template in KNOWN_TEMPLATES else template
    )
    env = Environment(autoescape=False)
    env.filters.setdefault("tojson", lambda v, **kw: json.dumps(v, default=str))
    tpl = env.from_string(template_src)
    msg_list = [
        m.model_dump(exclude_none=False) if isinstance(m, SFTMessage) else m for m in messages
    ]
    return tpl.render(messages=msg_list, add_generation_prompt=add_generation_prompt)
