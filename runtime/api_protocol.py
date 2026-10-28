"""OpenAI message normalization and incremental checkpoint output framing."""
import json
import re

from engine.tool_calls import openai_tool_calls,parse_tool_calls
from tool_schema import ToolSchemaError,ToolCallError,validate_tool_schema


class RequestError(ValueError):
    pass


THINKING_EFFORT_BUDGETS = dict(low=1024, medium=4096, high=16384)


def _thinking_options(body):
    template = body.get('chat_template_kwargs')
    if template is None:
        template = {}
    if not isinstance(template, dict):
        raise RequestError('Invalid chat_template_kwargs.')
    allowed = {'enable_thinking', 'reasoning_effort', 'thinking_budget',
        'thinking_budget_tokens', 'preserve_thinking'}
    unsupported = set(template) - allowed
    if unsupported:
        raise RequestError('Unsupported chat_template_kwargs: ' + ', '.join(sorted(unsupported)))

    def option(names, validate):
        values = []
        for source in (body, template):
            for name in names:
                if name in source:
                    value = source[name]
                    validate(value)
                    values.append(value)
        if values and any(value != values[0] for value in values[1:]):
            raise RequestError('Conflicting values for ' + '/'.join(names) + '.')
        return values[0] if values else None

    def boolean(value):
        if type(value) is not bool:
            raise RequestError('Thinking switches must be booleans.')

    def effort_value(value):
        if not isinstance(value, str) or value not in ('none', 'off', *THINKING_EFFORT_BUDGETS):
            raise RequestError('reasoning_effort must be none, off, low, medium, or high.')

    def budget_value(value):
        if type(value) is not int or value < 1:
            raise RequestError('thinking_budget/thinking_budget_tokens must be a positive integer.')

    enabled = option(('enable_thinking',), boolean)
    effort = option(('reasoning_effort',), effort_value)
    budget = option(('thinking_budget', 'thinking_budget_tokens'), budget_value)
    preserve = option(('preserve_thinking',), boolean)
    effort_enabled = None if effort is None else effort not in ('none', 'off')
    if enabled is not None and effort_enabled is not None and enabled != effort_enabled:
        raise RequestError('enable_thinking conflicts with reasoning_effort.')
    if budget is not None and (enabled is False or effort_enabled is False):
        raise RequestError('A thinking budget conflicts with disabled thinking.')
    thinking = (budget is not None or effort_enabled is True or enabled is True)
    if budget is None and effort in THINKING_EFFORT_BUDGETS:
        budget = THINKING_EFFORT_BUDGETS[effort]
    return dict(thinking=thinking, reasoning_effort=effort, thinking_budget=budget,
        preserve_thinking=preserve)


def _text(value,allow_none=False):
    if value is None and allow_none:return ''
    if isinstance(value,str):return value
    if isinstance(value,list):
        if all(isinstance(x,dict) and x.get('type') in ('text','input_text') and isinstance(x.get('text'),str) for x in value):
            return ''.join(x['text'] for x in value)
    raise RequestError('Only text message content is supported.')


def normalize_request(body,model,max_output=4096):
    if type(max_output) is not int or max_output<0:
        raise ValueError('Output limit must be zero or a positive integer.')
    if not isinstance(body,dict):raise RequestError('Expected a JSON object.')
    if body.get('model')!=model:raise RequestError(f'Use model {model}.')
    messages=body.get('messages')
    if not isinstance(messages,list) or not messages:raise RequestError('messages must be a nonempty array.')
    normalized=[]
    for item in messages:
        if not isinstance(item,dict):raise RequestError('Invalid message.')
        role=item.get('role')
        if role=='developer':role='system'
        if role not in ('system','user','assistant','tool'):raise RequestError('Unsupported message role.')
        message=dict(role=role,content=_text(item.get('content'),role=='assistant'))
        if 'name' in item:message['name']=item['name']
        if role=='tool':
            if not isinstance(item.get('tool_call_id'),str):raise RequestError('Tool messages need tool_call_id.')
            message['tool_call_id']=item['tool_call_id']
        if item.get('tool_calls'):
            if role!='assistant' or not isinstance(item['tool_calls'],list):raise RequestError('Invalid assistant tool_calls.')
            calls=[]
            for call in item['tool_calls']:
                if not isinstance(call,dict) or call.get('type')!='function':raise RequestError('Only function tools are supported.')
                function=call.get('function',{})
                if not isinstance(function,dict) or not isinstance(function.get('name'),str):raise RequestError('Invalid tool function.')
                arguments=function.get('arguments')
                if isinstance(arguments,str):
                    try:arguments=json.loads(arguments)
                    except ValueError as error:raise RequestError('Tool arguments must be valid JSON.') from error
                if not isinstance(arguments,dict):raise RequestError('Tool arguments must be a JSON object.')
                calls.append(dict(id=call.get('id',''),type='function',function=dict(name=function['name'],arguments=arguments)))
            message['tool_calls']=calls
        if isinstance(item.get('reasoning_content'),str):message['reasoning_content']=item['reasoning_content']
        normalized.append(message)
    tools=body.get('tools') or []
    if not isinstance(tools,list) or len(tools)>128:raise RequestError('Expected at most 128 function tools.')
    names=set()
    for index,tool in enumerate(tools):
        if not isinstance(tool,dict) or tool.get('type')!='function':raise RequestError('Only function tools are supported.')
        function=tool.get('function',{})
        if not isinstance(function,dict) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',function.get('name','')):
            raise RequestError('Invalid function name.')
        if function['name'] in names:raise RequestError('Duplicate function name.')
        names.add(function['name'])
        validate_tool_schema(function.get('parameters',{}), tool=function['name'],
                             param=f'tools[{index}].function.parameters')
    choice=body.get('tool_choice','auto')
    if choice=='none':tools=[]
    elif choice!='auto':raise RequestError('tool_choice currently supports auto or none.')
    # This accepted MTP path is greedy; do not silently pretend to sample.
    temperature=body.get('temperature',0)
    if temperature is None:temperature=0
    if isinstance(temperature,bool) or temperature!=0:raise RequestError('This trial uses greedy generation. Set temperature to 0.')
    if body.get('n',1)!=1:raise RequestError('Only n=1 is supported.')
    for key in ('presence_penalty','frequency_penalty'):
        if body.get(key,0) not in (None,0):raise RequestError(f'{key} is not supported.')
    if body.get('logprobs') or body.get('top_logprobs'):raise RequestError('logprobs is not supported.')
    if body.get('stop'):raise RequestError('Custom stop strings are not supported in this trial.')
    response_format=body.get('response_format') or {}
    if not isinstance(response_format,dict):raise RequestError('Invalid response_format.')
    if response_format.get('type','text')!='text':raise RequestError('Structured response_format is not supported; specify the format in your prompt.')
    # Zero disables the fixed output cap. The runtime resolves omitted/null
    # budgets after applying the chat template and tokenizing the prompt.
    default_budget=None if max_output==0 else 1024
    budget=body.get('max_completion_tokens',body.get('max_tokens',default_budget))
    if budget is not None or max_output:
        if type(budget) is not int or budget<1 or (max_output and budget>max_output):
            limit=f'1..{max_output}' if max_output else 'a positive integer or null'
            raise RequestError(f'max_tokens must be {limit}.')
    thinking_options=_thinking_options(body)
    if not isinstance(body.get('stream',False),bool):raise RequestError('stream must be a boolean.')
    stream_options=body.get('stream_options') or {}
    if not isinstance(stream_options,dict):raise RequestError('Invalid stream_options.')
    return dict(messages=normalized,tools=tools,max_tokens=budget,**thinking_options,
                stream=body.get('stream',False),include_usage=bool(stream_options.get('include_usage')))


class OutputFramer:
    """Hold partial control tags; never stream raw tool XML as chat content."""
    markers={'<think>':'reasoning','</think>':'content','<tool_call>':'tool',
             '<|im_end|>':'end','<|endoftext|>':'end','<|im_start|>':'end'}

    def __init__(self,tools,emit=None,*,starts_in_reasoning=False):
        if type(starts_in_reasoning) is not bool:
            raise ValueError('starts_in_reasoning must be a boolean')
        self.tools=tools; self.emit=emit or (lambda delta:None)
        self.offset=0; self.raw=''; self.mode='reasoning' if starts_in_reasoning else 'content'; self.tool_start=None
        self.content=''; self.reasoning=''; self.calls=[]

    def _emit_text(self,value):
        if not value or self.mode=='end':return
        if self.mode=='reasoning':
            self.reasoning+=value; self.emit(dict(reasoning_content=value))
        else:
            self.content+=value; self.emit(dict(content=value))

    def feed(self,raw,final=False):
        if not raw.startswith(self.raw):raise RuntimeError('Non-monotonic detokenization.')
        self.raw=raw
        while self.offset<len(raw):
            if self.mode=='end':self.offset=len(raw); break
            if self.mode=='tool':
                end=raw.find('</tool_call>',self.offset)
                if end<0:
                    if final:raise ToolCallError('Model produced an incomplete tool call')
                    break
                end+=len('</tool_call>')
                calls=openai_tool_calls(parse_tool_calls(raw[self.tool_start:end],self.tools))
                for call in calls:
                    index=len(self.calls); self.calls.append(call)
                    self.emit(dict(tool_calls=[dict(index=index,**call)]))
                self.offset=end; self.mode='content'; continue
            found=[(raw.find(marker,self.offset),marker,mode) for marker,mode in self.markers.items()]
            found=[item for item in found if item[0]>=0]
            if found:
                position,marker,mode=min(found,key=lambda item:item[0])
                self._emit_text(raw[self.offset:position])
                self.offset=position+len(marker); self.mode=mode
                if mode=='tool':self.tool_start=position
                continue
            end=len(raw)
            if not final or self.mode=='reasoning':
                # A token can end in '<tool_' or the opening bytes of a tag.
                # A length-limited open reasoning block must not leak a
                # truncated closing/control tag into the returned text.
                for marker in self.markers:
                    for size in range(1,min(len(marker),len(raw)-self.offset+1)):
                        if raw.endswith(marker[:size]):end=min(end,len(raw)-size)
            self._emit_text(raw[self.offset:end]); self.offset=end
            break
        if final and self.mode=='tool':raise ToolCallError('Model produced an incomplete tool call')

    def message(self):
        result=dict(role='assistant',content=self.content if self.content else (None if self.calls else ''))
        if self.reasoning:result['reasoning_content']=self.reasoning
        if self.calls:result['tool_calls']=self.calls
        return result
