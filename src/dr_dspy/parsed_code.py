from __future__ import annotations

import ast
import copy
import io
import tokenize
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Variable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    var_type: str | None = None


class FunctionSignature(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tree: ast.AST = Field(exclude=True)
    code_str: str = ""
    signature_str: str = ""
    function_name: str = ""
    function_args: list[Variable] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def derive_signature(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        values: dict[str, Any] = {
            str(key): value for key, value in data.items()
        }
        tree = values.get("tree")
        code_str = values.get("code_str")
        if tree is None and code_str is not None:
            tree = ast.parse(str(code_str))
        if tree is None:
            return values
        node = find_function_node(tree)
        values.setdefault("tree", tree)
        values.setdefault(
            "code_str",
            str(code_str) if code_str is not None else ast.unparse(tree),
        )
        if node is not None:
            values.setdefault("signature_str", format_function_signature(node))
            values.setdefault("function_name", node.name)
            values.setdefault("function_args", extract_function_args(node))
        return values


class ParsedCode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    code_str: str
    tree: ast.AST | None = Field(default=None, exclude=True)
    signature: FunctionSignature | None = None
    signatures: list[FunctionSignature] = Field(default_factory=list)
    code_without_comments: str = ""
    comments: str = ""
    display_title: str = "ParsedCode"

    @model_validator(mode="before")
    @classmethod
    def derive_from_code_str(cls, data: object) -> object:
        if not isinstance(data, dict) or "code_str" not in data:
            return data
        values: dict[str, Any] = {
            str(key): value for key, value in data.items()
        }
        code_str = str(values["code_str"])
        tree = ast.parse(code_str)
        signatures = extract_function_signatures(tree=tree)
        values.setdefault("tree", tree)
        if signatures:
            values.setdefault("signature", signatures[0])
        values.setdefault("signatures", signatures)
        values.setdefault(
            "code_without_comments",
            remove_comments(tree, remove_docstrings=True),
        )
        values.setdefault("comments", collect_comments(code_str, tree))
        return values


def ensure_tree(
    code_str: str | None = None,
    tree: ast.AST | None = None,
) -> ast.AST:
    if code_str is None and tree is None:
        raise ValueError("Need code_str or parse tree")
    if tree is None:
        if code_str is None:
            raise ValueError("Need code_str when tree is not provided")
        tree = ast.parse(code_str)
    return tree


def format_function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({args}){returns}:"


def find_function_node(
    tree: ast.AST,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    if isinstance(tree, ast.FunctionDef | ast.AsyncFunctionDef):
        return tree
    return next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        ),
        None,
    )


def extract_function_args(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[Variable]:
    args = node.args
    all_args = args.posonlyargs + args.args + args.kwonlyargs
    return [
        Variable(
            name=arg.arg,
            var_type=ast.unparse(arg.annotation) if arg.annotation else None,
        )
        for arg in all_args
    ]


def extract_function_signatures(
    code_str: str | None = None,
    tree: ast.AST | None = None,
) -> list[FunctionSignature]:
    resolved_tree = ensure_tree(code_str, tree)
    return [
        FunctionSignature(tree=node)
        for node in ast.walk(resolved_tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def is_string_literal_stmt(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def strip_leading_docstring(
    node: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> None:
    body = node.body
    if body and is_string_literal_stmt(body[0]):
        node.body = body[1:] or [ast.Pass()]


def remove_comments(tree: ast.AST, *, remove_docstrings: bool = True) -> str:
    if remove_docstrings:
        tree = copy.deepcopy(tree)
        for node in ast.walk(tree):
            if isinstance(
                node,
                (
                    ast.Module
                    | ast.FunctionDef
                    | ast.AsyncFunctionDef
                    | ast.ClassDef
                ),
            ):
                strip_leading_docstring(node)
    return ast.unparse(tree)


def extract_hash_comments(code_str: str) -> list[tuple[int, str]]:
    tokens = tokenize.generate_tokens(io.StringIO(code_str).readline)
    return [
        (tok.start[0], tok.string[1:].strip())
        for tok in tokens
        if tok.type == tokenize.COMMENT
    ]


def extract_docstrings(tree: ast.AST) -> list[tuple[int, str]]:
    docstrings: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        ):
            continue
        if node.body and is_string_literal_stmt(node.body[0]):
            string_node = node.body[0]
            if (
                isinstance(string_node, ast.Expr)
                and isinstance(string_node.value, ast.Constant)
                and isinstance(string_node.value.value, str)
            ):
                docstrings.append(
                    (string_node.lineno, string_node.value.value)
                )
    return docstrings


def collect_comments(code_str: str, tree: ast.AST) -> str:
    items = extract_hash_comments(code_str) + extract_docstrings(tree)
    return "\n".join(
        text for _lineno, text in sorted(items, key=lambda item: item[0])
    )
