"""Single-sequence chat token IDs across tokenizer return conventions."""
from collections.abc import Mapping
import json
from pathlib import Path


def generation_eos_ids(tokenizer,text_config,model_dir):
    """Honor the checkpoint's generation EOS list and chat tokenizer EOS."""
    values=[text_config.get('eos_token_id'),getattr(tokenizer,'eos_token_id',None)]
    path=Path(model_dir)/'generation_config.json'
    if path.exists():values.append(json.loads(path.read_text()).get('eos_token_id'))
    output=set()
    for value in values:
        if value is None:continue
        for token in value if isinstance(value,(list,tuple)) else [value]:
            if not isinstance(token,int) or isinstance(token,bool) or token<0:
                raise ValueError('Invalid checkpoint EOS token ID')
            output.add(token)
    if not output:raise ValueError('Checkpoint has no generation EOS token')
    return output


def chat_token_ids(tokenizer,messages,thinking=False,tools=None,*,preserve_thinking=None):
    options={} if tools is None else {'tools':tools}
    if preserve_thinking is not None:
        if type(preserve_thinking) is not bool:
            raise ValueError('preserve_thinking must be a boolean or None')
        options['preserve_thinking']=preserve_thinking
    encoded=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        enable_thinking=thinking,return_dict=False,**options)
    # Some checkpoint wrappers still return BatchEncoding even when the
    # caller requests a list. Do not iterate its string dictionary keys.
    if isinstance(encoded,Mapping):encoded=encoded['input_ids']
    if hasattr(encoded,'tolist'):encoded=encoded.tolist()
    if not isinstance(encoded,(list,tuple)) or not encoded:
        raise ValueError('Chat template must return one nonempty token sequence')
    if isinstance(encoded[0],(list,tuple)):
        raise ValueError('This engine accepts one chat sequence per request')
    return [int(token) for token in encoded]


def prompt_starts_in_reasoning(tokenizer,ids,*,tail_tokens=128):
    """Inspect only the actual tokenized assistant prefix; never alter IDs.

    A user's literal <think> tag or an older assistant turn does not determine
    the initial generated-output state. The active final assistant prefix must
    itself contain an opening <think> with no following closing/end marker.
    """
    if type(tail_tokens) is not int or tail_tokens < 1:
        raise ValueError('tail_tokens must be a positive integer')
    if not ids:
        return False
    tail=tokenizer.decode(ids[-tail_tokens:],skip_special_tokens=False,
        clean_up_tokenization_spaces=False)
    assistant_marker='<|im_start|>assistant'
    start=tail.rfind(assistant_marker)
    if start<0 or start!=tail.rfind('<|im_start|>'):
        return False
    suffix=tail[start+len(assistant_marker):]
    if any(marker in suffix for marker in ('<|im_end|>','<|endoftext|>')):
        return False
    opening=suffix.rfind('<think>')
    return opening>=0 and opening>suffix.rfind('</think>')
