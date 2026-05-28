from tree_sitter import Language, Parser
import json
import os 
import warnings
import tqdm
import random
from data_utils import is_valid_identifier
from data_utils import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)

path = '/root/autodl-tmp/SLODA/dataset/parse/my-languages.so'
c_path = '/root/autodl-tmp/SLODA/dataset/parse/tree-sitter-c'
cpp_path = '/root/autodl-tmp/SLODA/dataset/parse/tree-sitter-cpp'
# 构建和加载C/C++语法
Language.build_library(
  path,
  [c_path, cpp_path]
)

C_LANGUAGE = Language(path, 'c')
CPP_LANGUAGE = Language(path, 'cpp')
warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")

PARSE_DIR = '/root/autodl-tmp/SLODA/dataset/parse'
LIB_PATH = path
C_GRAMMAR_PATH = c_path
CPP_GRAMMAR_PATH = cpp_path

LANGUAGE_C = C_LANGUAGE
LANGUAGE_CPP = CPP_LANGUAGE


def create_parser(language_name='c'):
    parser = Parser()
    if language_name == 'c':
        parser.set_language(LANGUAGE_C)
    elif language_name == 'cpp':
        parser.set_language(LANGUAGE_CPP)
    else:
        raise ValueError("Unsupported language")
    return parser


def map_node_to_category(node_type: str, node_text: str) -> str:
    if node_type in ('if_statement', 'for_statement', 'while_statement', 'do_statement', 'switch_statement'):
        return 'Control Flow Statements'
    elif node_type == 'call_expression':
        return 'Function Calls'
    elif node_type == 'identifier':
        return 'Identifiers'
    elif node_type == 'assignment_expression':
        return 'Assignments'
    elif node_type == 'declaration':
        if '=' in node_text:
            return 'Assignments'
        else:
            return 'Declarations'
    elif node_type == 'string_literal':
        return 'String Literals'
    elif node_type == 'number_literal':
        return 'Numeric Literals'
    elif node_type == 'char_literal':
        return 'Character Literals'
    else:
        return node_type


def build_text_category_map(code: str, language=None):
    if language is None or language == 'auto':
        parser = Parser()
        parser.set_language(LANGUAGE_CPP)
        tree = parser.parse(bytes(code, 'utf8'))
        if tree.root_node.has_error:
            parser.set_language(LANGUAGE_C)
            tree = parser.parse(bytes(code, 'utf8'))
    else:
        parser = create_parser(language)
        tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    mappings = []

    def walk(node):
        start, end = node.start_byte, node.end_byte
        text = code[start:end]
        cat = map_node_to_category(node.type, text)
        mappings.append((text, cat, node.start_byte, node.end_byte))
        for child in node.children:
            walk(child)

    walk(root_node)
    return mappings


def query_code_type(code: str, target: str, language=None) -> str:
    mappings = build_text_category_map(code, language)

    for text, cat, _, _ in mappings:
        if text == target:
            return cat

    candidates = []
    meaningful_categories = {
        'Control Flow Statements', 'Function Calls', 'Assignments',
        'Identifiers', 'String Literals', 'Numeric Literals', 'Character Literals'
    }

    for text, cat, start, end in mappings:
        if target in text:
            length = end - start
            is_meaningful = cat in meaningful_categories
            candidates.append((0 if is_meaningful else 1, length, cat))

    if candidates:
        candidates.sort()
        return candidates[0][2]

    return ''

def get_identifiers(code):
    def parse_code(language):
        parser = Parser()
        parser.set_language(language)
        return parser.parse(bytes(code, "utf8"))

    def extract_identifiers(node):
        identifiers = []
        if node.type == 'identifier':
            identifiers.append(node.text.decode('utf8'))
        for child in node.children:
            identifiers.extend(extract_identifiers(child))
        return identifiers

    # 尝试使用C++解析器
    tree = parse_code(CPP_LANGUAGE)
    if tree.root_node.has_error:
        # 如果C++解析失败，尝试使用C解析器
        tree = parse_code(C_LANGUAGE)

    return extract_identifiers(tree.root_node)

def get_code_tokens(code):
    parser = Parser()
    parser.set_language(CPP_LANGUAGE)
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node

    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    # print(code)
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    return code_tokens

def get_example(code, tgt_word, substitute):
    parser = Parser()
    parser.set_language(CPP_LANGUAGE)
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    replace_pos = {}
    for index, code_token in enumerate(code_tokens):
        if code_token == tgt_word:
            try:
                replace_pos[tokens_index[index][0][0]].append((tokens_index[index][0][1], tokens_index[index][1][1]))
            except:
                replace_pos[tokens_index[index][0][0]] = [(tokens_index[index][0][1], tokens_index[index][1][1])]
    diff = len(substitute) - len(tgt_word)
    for line in replace_pos.keys():
        for index, pos in enumerate(replace_pos[line]):
            code[line] = code[line][:pos[0]+index*diff] + substitute + code[line][pos[1]+index*diff:]

    return "\n".join(code)

def get_function_call(code):
    """
    从给定的 C/C++ 代码片段中提取所有被调用的函数名。
    返回一个去重的函数名列表（保留顺序）。
    """
    def parse_code(language):
        parser = Parser()
        parser.set_language(language)
        return parser.parse(bytes(code, "utf8"))

    def extract_function_calls(node):
        calls = []
        # 如果是函数调用表达式
        if node.type == 'call_expression':
            # 在 Tree-sitter 的 C/C++ 语法中，函数名通常位于 'function' 字段
            if 'function' in node.child_by_field_name:
                func_node = node.child_by_field_name('function')
                # 处理普通函数调用（如 foo()）
                if func_node.type == 'identifier':
                    calls.append(func_node.text.decode('utf8'))
                # 处理成员函数调用（如 obj.method()），此时 function 是 field_expression
                elif func_node.type == 'field_expression':
                    # 取 field 部分（即 method 名）
                    field_node = func_node.child_by_field_name('field')
                    if field_node and field_node.type == 'identifier':
                        calls.append(field_node.text.decode('utf8'))
                # 支持函数指针调用等复杂情况可继续扩展
        # 递归遍历子节点
        for child in node.children:
            calls.extend(extract_function_calls(child))
        return calls

    # 尝试用 C++ 解析
    tree = parse_code(CPP_LANGUAGE)
    if tree.root_node.has_error:
        tree = parse_code(C_LANGUAGE)

    raw_calls = extract_function_calls(tree.root_node)

    # 去重但保留顺序
    seen = set()
    unique_calls = []
    for name in raw_calls:
        if name not in seen:
            seen.add(name)
            unique_calls.append(name)

    return unique_calls

def replace_function_name(code: str, func_name: str, substitute: str) -> str:
    """
    替换 C/C++ 代码中所有函数定义和声明的函数名为 substitute。
    
    Args:
        code (str): 原始代码字符串
        func_name (str): 要被替换的原函数名
        substitute (str): 新的函数名
    
    Returns:
        str: 替换后的代码
    """
    from tree_sitter import Parser

    parser = Parser()
    parser.set_language(CPPLANGUAGE)  # 确保你已定义 CPP_LANGUAGE

    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node

    # 构建查询：匹配函数定义和函数声明中的函数名
    # 注意：C 和 C++ 的语法略有不同，但 function_declarator + identifier 是共通的
    query = CPP_LANGUAGE.query("""
    ; 函数定义：如 void foo() { }
    (function_definition
      declarator: (function_declarator
        declarator: (identifier) @func_id
      )
    )

    ; 函数声明（无函数体）：如 void foo();
    (declaration
      declarator: (function_declarator
        declarator: (identifier) @func_id
      )
    )

    ; C++ 类中的成员函数声明（在 class_specifier 内）
    (field_declaration
      declarator: (function_declarator
        declarator: (identifier) @func_id
      )
    )
    """)

    # 收集所有匹配的 identifier 节点
    matches = query.captures(root_node)
    edit_ranges = []

    for node, _ in matches:
        if node.type == 'identifier':
            name_text = node.text.decode('utf8')
            if name_text == func_name:
                edit_ranges.append((node.start_byte, node.end_byte))

    # 从后往前替换，避免字节偏移变化影响
    code_bytes = bytearray(code, 'utf8')
    substitute_bytes = substitute.encode('utf8')

    for start, end in sorted(edit_ranges, reverse=True):
        code_bytes[start:end] = substitute_bytes

    return code_bytes.decode('utf8')


def get_example_batch(code, chromesome):
    parser = Parser()
    code = code.replace("\\n", "\n")
    parser.set_language(CPP_LANGUAGE)
    tree = parser.parse(bytes(code, 'utf8'))
    if tree.root_node.has_error:
        # 如果C++解析失败，尝试使用C解析器
        parser.set_language(C_LANGUAGE)
        tree = parser.parse(bytes(code, 'utf8'))

    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    replace_pos = {}
    for tgt_word in chromesome.keys():
        diff = len(chromesome[tgt_word]) - len(tgt_word)
        for index, code_token in enumerate(code_tokens):
            if code_token == tgt_word:
                try:
                    replace_pos[tokens_index[index][0][0]].append((tgt_word, chromesome[tgt_word], diff, tokens_index[index][0][1], tokens_index[index][1][1]))
                except:
                    replace_pos[tokens_index[index][0][0]] = [(tgt_word, chromesome[tgt_word], diff, tokens_index[index][0][1], tokens_index[index][1][1])]
    for line in replace_pos.keys():
        diff = 0
        for index, pos in enumerate(replace_pos[line]):
            code[line] = code[line][:pos[3]+diff] + pos[1] + code[line][pos[4]+diff:]
            diff += pos[2]

    return "\n".join(code)


def change_code(code, identifiers, code_vocab):
    number_of_elements = len(identifiers) // 5 if len(identifiers) >= 5 else 1
    selected_elements = random.sample(identifiers, number_of_elements)
    result_dict = {key: random.choice(code_vocab) for key in selected_elements}
    new_code = get_example_batch(code, result_dict)
    return new_code, result_dict
