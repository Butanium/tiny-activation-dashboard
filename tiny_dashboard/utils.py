import inspect
import ast
import sqlite3
import json


def parse_list_str(s: str) -> list[int]:
    s = s.strip()
    if not s.startswith("["):
        s = "[" + s
    if not s.endswith("]"):
        s = s + "]"
    return ast.literal_eval(s)


def apply_chat(
    text: str, tokenizer, add_bos: bool = True, enable_thinking: bool = None
) -> str:
    """Apply chat formatting to text using the tokenizer"""
    splitted = text.split("<eot>")
    is_user = True
    chat = []
    for s in splitted[:-1]:
        role = "user" if is_user else "assistant"
        chat.append({"role": role, "content": s})
        is_user = not is_user
    if is_user:
        chat.append({"role": "user", "content": splitted[-1]})
    len_bos = len(tokenizer.bos_token) if tokenizer.bos_token is not None else 0
    # Check if tokenizer supports enable_thinking parameter
    apply_chat_params = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        apply_chat_params["enable_thinking"] = enable_thinking
    formated_chat = tokenizer.apply_chat_template(chat, **apply_chat_params)[
        0 if add_bos else len_bos :
    ]
    if not is_user:
        formated_chat += splitted[-1]
    return formated_chat


def sanitize_html_content(s: str) -> str:
    """
    Sanitize a string to be used as HTML content.
    """
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("'", "&apos;")
        .replace('"', "&quot;")
    )


def sanitize_token(
    token: str, non_breaking_space: bool = True, keep_newline: bool = True
) -> str:
    return (
        sanitize_html_content(token)
        .replace("Ċ", "\n")
        .replace("\n", "\\n<br>" if keep_newline else "\\n")
        .replace("▁", " ")
        .replace("Ġ", " ")
        .replace(" ", "&nbsp;" if non_breaking_space else " ")
    )


def sanitize_tokens(
    tokens: list[str], non_breaking_space: bool = True, keep_newline: bool = True
) -> list[str]:
    return [sanitize_token(t, non_breaking_space, keep_newline) for t in tokens]


def update_string(s: str, str_map: dict[str, str]) -> str:
    """Update a string with a mapping from old strings to new strings."""
    for old, new in str_map.items():
        s = s.replace(old, new)
    return s


def update_template_string(s: str, str_map: dict[str, str]) -> str:
    """Update a template string with a mapping from old strings to new strings."""
    return update_string(s, {"{{" + k + "}}": v for k, v in str_map.items()})


class DummyModel:
    def __getattr__(self, name):
        if "__" in name:
            return super().__getattribute__(name)
        raise ValueError(
            f"Attempted to access '{name}' on a DummyModel instance, which is intended solely as a placeholder."
        )

    def __getattribute__(self, name):
        if "__" in name:
            return super().__getattribute__(name)
        raise ValueError(
            f"Attempted to access '{name}' on a DummyModel instance, which is intended solely as a placeholder."
        )

    def __call__(self, *args, **kwargs):
        raise ValueError(
            "Attempted to call a DummyModel instance, which is intended solely as a placeholder."
        )


class LazyReadDict:
    def __init__(self, db_path, column_name: str, table_name="data_table"):
        self.db_path = db_path
        self.column_name = column_name
        self.table_name = table_name
        self._init_keys()

    def _init_keys(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT key FROM {self.table_name}")
            self._keys = [row[0] for row in cursor.fetchall()]

    def __getitem__(self, key):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT {self.column_name} FROM {self.table_name} WHERE key = ?",
                (key,),
            )
            rows = cursor.fetchall()

            if not rows:
                raise KeyError(key)

            if len(rows) > 1:
                raise ValueError(f"Multiple entries found for key {key}")
            return json.loads(rows[0][0])

    def keys(self):
        return self._keys

    def __contains__(self, key):
        return key in self._keys
