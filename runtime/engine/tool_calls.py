"""Schema-checked complete tool calls in this checkpoint's chat grammar.

This parser returns data only. It never executes a tool or evaluates code.
The grammar comes from the actual checkpoint template archived in diagnostics.
"""
import json
import re
import uuid

from tool_schema import ToolCallError, validate_tool_arguments


def _decode(raw,schema,*,tool,param,root):
    # References are relative to the complete parameters document, never the
    # extracted property. Admission already checked targets and rejected cycles.
    while isinstance(schema,dict) and '$ref' in schema:
        ref=schema['$ref'];schema=root
        for key in ref[2:].split('/') if ref!='#' else ():
            key=key.replace('~1','/').replace('~0','~')
            schema=schema[int(key)] if isinstance(schema,list) else schema[key]
    if isinstance(schema,dict) and schema.get('type')=='string':
        # Remove the template's one framing newline, preserving value spaces.
        if raw.startswith('\n'):raw=raw[1:]
        if raw.endswith('\n'):raw=raw[:-1]
        value=raw
    else:
        def invalid_constant(value):
            raise ToolCallError('Nonfinite JSON tool argument',tool=tool,param=param)
        try:
            value=json.loads(raw.strip(),parse_constant=invalid_constant)
        except json.JSONDecodeError as error:
            raise ToolCallError('Tool argument must be valid JSON',tool=tool,param=param) from error
    return value


def parse_tool_calls(text,tools):
    available={t['function']['name']:t['function'] for t in tools}
    blocks=re.findall(r'<tool_call>(.*?)</tool_call>',text,re.DOTALL)
    if text.count('<tool_call>')!=len(blocks) or text.count('</tool_call>')!=len(blocks):
        raise ToolCallError('Incomplete tool-call block')
    result=[]
    for block in blocks:
        match=re.fullmatch(r'\s*<function=([^>]+)>(.*?)</function>\s*',block,re.DOTALL)
        if match is None:raise ToolCallError('Invalid checkpoint tool-call grammar')
        name,body=match.groups()
        if name not in available:raise ToolCallError('Model requested an unavailable tool',tool=name)
        function=available[name]; schema=function.get('parameters',{})
        properties=schema.get('properties',{})
        arguments={}; consumed=0
        for parameter in re.finditer(r'<parameter=([^>]+)>(.*?)</parameter>',body,re.DOTALL):
            if body[consumed:parameter.start()].strip():raise ToolCallError('Unexpected tool-call content',tool=name)
            key,raw=parameter.groups()
            if key in arguments:raise ToolCallError('Duplicate tool argument',tool=name,param='arguments.'+key)
            patterns=[s for p,s in schema.get('patternProperties',{}).items() if re.search(p,key)]
            field_schema=properties.get(key,patterns[0] if patterns else schema.get('additionalProperties',{}))
            if field_schema is False:
                raise ToolCallError('Unknown tool argument',tool=name,param='arguments.'+key)
            arguments[key]=_decode(raw,field_schema,tool=name,param='arguments.'+key,root=schema); consumed=parameter.end()
        if body[consumed:].strip():raise ToolCallError('Unexpected trailing tool argument content',tool=name)
        validate_tool_arguments(arguments,schema,tool=name)
        result.append(dict(id='call_'+uuid.uuid4().hex[:24],type='function',
                           function=dict(name=name,arguments=arguments)))
    return result


def openai_tool_calls(calls):
    return [dict(call,function=dict(name=call['function']['name'],
                 arguments=json.dumps(call['function']['arguments'],ensure_ascii=False))) for call in calls]
