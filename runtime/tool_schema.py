"""Offline tool-schema admission and value-free, structured protocol errors."""
import math
import re

from jsonschema import Draft202012Validator, SchemaError
from jsonschema.validators import validator_for
from referencing import Registry


def safe_path(value):
    # Names and paths only: never include model text, schema values or arguments.
    return re.sub(r'[^A-Za-z0-9_.\[\]-]', '?', str(value))[:240]


class ToolSchemaError(ValueError):
    code = 'invalid_tool_schema'

    def __init__(self, message, *, tool, param):
        self.tool = safe_path(tool)
        self.param = safe_path(param)
        super().__init__(f"Invalid parameters schema for tool '{self.tool}' at {self.param}: {message}")


class ToolCallError(ValueError):
    code = 'invalid_tool_call'

    def __init__(self, message, *, tool=None, param=None):
        self.tool = safe_path(tool) if tool is not None else None
        self.param = safe_path(param) if param is not None else None
        location = f" for tool '{self.tool}'" if self.tool else ''
        if self.param:
            location += f' at {self.param}'
        super().__init__(message + location + '.')


def _validator(schema):
    # An explicit empty registry prevents remote reference retrieval / SSRF.
    return validator_for(schema, default=Draft202012Validator)(schema, registry=Registry())


def _children(schema):
    if not isinstance(schema, dict):
        return
    for keyword in ('properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas'):
        for key, child in schema.get(keyword, {}).items():
            yield (keyword, key), child
    for keyword in ('items', 'additionalProperties', 'unevaluatedProperties', 'unevaluatedItems',
                    'contains', 'propertyNames', 'not', 'if', 'then', 'else', 'additionalItems'):
        child = schema.get(keyword)
        if isinstance(child, list):
            for i, item in enumerate(child):
                yield (keyword, i), item
        elif isinstance(child, (dict, bool)):
            yield (keyword,), child
    for keyword in ('allOf', 'anyOf', 'oneOf', 'prefixItems'):
        for i, child in enumerate(schema.get(keyword, [])):
            yield (keyword, i), child
    for key, child in schema.get('dependencies', {}).items():
        if isinstance(child, (dict, bool)):
            yield ('dependencies', key), child


def validate_tool_schema(schema, *, tool, param):
    def fail(message, suffix=()):
        path = param + ''.join(f'[{p}]' if isinstance(p, int) else '.' + p for p in suffix)
        raise ToolSchemaError(message, tool=tool, param=path)

    # Empty/missing parameters remain compatible with existing no-argument tools.
    if not isinstance(schema, dict) or (schema and schema.get('type') != 'object'):
        fail('Expected a JSON Schema object with type="object" and a properties map; '
             'flat parameter maps are not supported.')
    pending = [(schema, 0)]
    nodes = 0
    while pending:
        node, depth = pending.pop()
        nodes += 1
        if nodes > 10000 or depth > 64:
            fail('Schema exceeds the supported complexity limit.')
        if isinstance(node, (dict, list)):
            pending.extend((child, depth + 1) for child in
                           (node.values() if isinstance(node, dict) else node))
    try:
        validator_for(schema, default=Draft202012Validator).check_schema(schema)
    except SchemaError as error:
        fail('Invalid JSON Schema keyword shape; required must be an array of property names, '
             'and properties/items must contain valid schemas.', tuple(error.absolute_path))

    count = 0
    def walk(node, path=(), active=frozenset()):
        nonlocal count
        count += 1
        if count > 10000 or len(path) > 64:
            fail('Schema exceeds the supported complexity limit.', path)
        if not isinstance(node, dict):
            return
        if id(node) in active:
            fail('Recursive schema references are not supported.', path)
        active = active | {id(node)}
        # Embedded resource IDs change reference scope; do not silently resolve
        # such a reference against the wrong root document.
        if '$id' in node and path:
            fail('Nested $id resources are not supported.', path + ('$id',))
        if '$dynamicRef' in node or '$recursiveRef' in node:
            fail('Dynamic schema references are not supported.', path)
        if '$ref' in node:
            ref = node['$ref']
            if ref != '#' and not ref.startswith('#/'):
                fail('Only local JSON Pointer references are supported.', path + ('$ref',))
            target = schema
            try:
                for key in ref[2:].split('/') if ref != '#' else ():
                    key = key.replace('~1', '/').replace('~0', '~')
                    target = target[int(key)] if isinstance(target, list) else target[key]
                if not isinstance(target, (dict, bool)):
                    raise KeyError()
            except (KeyError, IndexError, TypeError, ValueError):
                fail('Local schema reference does not resolve to a schema.', path + ('$ref',))
            walk(target, path + ('$ref',), active)
        for suffix, child in _children(node):
            walk(child, path + suffix, active)
    walk(schema)


def validate_tool_arguments(value, schema, *, tool, param='arguments'):
    def finite(item, path):
        if isinstance(item, float) and not math.isfinite(item):
            raise ToolCallError('Nonfinite JSON tool argument', tool=tool, param=path)
        if isinstance(item, dict):
            for key, child in item.items():
                finite(child, path + '.' + safe_path(key))
        elif isinstance(item, list):
            for i, child in enumerate(item):
                finite(child, path + f'[{i}]')
    finite(value, param)
    error = next(_validator(schema).iter_errors(value), None)
    if error is None:
        return
    path = param + ''.join(f'[{p}]' if isinstance(p, int) else '.' + safe_path(p)
                          for p in error.absolute_path)
    reason = {'type': 'Tool argument has wrong type', 'required': 'Missing required tool argument',
              'additionalProperties': 'Unknown tool argument', 'enum': 'Tool argument is outside enum'}
    if error.validator == 'required':
        missing = next((key for key in error.validator_value if key not in error.instance), None)
        if missing is not None:
            path += '.' + safe_path(missing)
    elif error.validator == 'additionalProperties' and isinstance(error.instance, dict):
        unknown = next((key for key in error.instance if key not in error.schema.get('properties', {})
                        and not any(re.search(p, key) for p in error.schema.get('patternProperties', {}))), None)
        if unknown is not None:
            path += '.' + safe_path(unknown)
    raise ToolCallError(reason.get(error.validator, 'Tool arguments do not satisfy JSON Schema ('
                                  + safe_path(error.validator) + ')'), tool=tool, param=path)
