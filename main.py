import sys
from collections import defaultdict
from typing import List
from dataclasses import dataclass

from lark import Lark, ast_utils, Transformer, v_args
from lark.tree import Meta

this_module = sys.modules[__name__]


class _Ast(ast_utils.Ast):
    # This will be skipped by create_transformer(), because it starts with an underscore
    pass


class _Statement(_Ast):
    # This will be skipped by create_transformer(), because it starts with an underscore
    pass


@dataclass
class Value(_Ast, ast_utils.WithMeta):
    "Uses WithMeta to include line-number metadata in the meta attribute"
    meta: Meta
    value: object


@dataclass
class Name(_Ast):
    name: str


@dataclass
class CodeBlock(_Ast, ast_utils.AsList):
    # Corresponds to code_block in the grammar
    statements: List[_Statement]


@dataclass
class IfBlock(_Statement):
    cond: Value
    then: CodeBlock


@dataclass
class Expression(_Statement):
    # Corresponds to set_var in the grammar
    value: object


@dataclass
class VariableAssignment(_Statement):
    # Corresponds to set_var in the grammar
    name: str
    value: Expression


@dataclass
class LeftExpression(_Statement):
    # Corresponds to set_var in the grammar
    expression: Expression


@dataclass
class RightExpression(_Statement):
    # Corresponds to set_var in the grammar
    expression: Expression


@dataclass
class UnaryOperator(_Ast):
    unary_operator: str


@dataclass
class BooleanValue(_Ast):
    value: str


@dataclass
class StringValue(_Ast):
    value: str

@dataclass
class NumberValue(_Ast):
    value: str


@dataclass
class VariableValue(_Ast):
    value: str


@dataclass
class BinaryOperator(_Ast):
    value: str


@dataclass
class BinaryExpression(_Statement):
    # Corresponds to set_var in the grammar
    left_expression: Expression
    operator: BinaryOperator
    right_expression: Expression


@dataclass
class UnaryExpression(_Statement):
    # Corresponds to set_var in the grammar
    operator: UnaryOperator
    expression: Expression


@dataclass
class Print(_Statement):
    value: Value

@dataclass
class ToString(_Statement):
    expression: Expression


class ToAst(Transformer):
    # Define extra transformation functions, for rules that don't correspond to an AST class.

    def STRING(self, s):
        # Remove quotation marks
        return s[1:-1]

    def DEC_NUMBER(self, n):
        return int(n)

    @v_args(inline=True)
    def start(self, x):
        return x


#
#   Define Parser
#

grammar = """
    start: code_block

    code_block: statement+

    ?statement: if_block | variable_assignment | print | expression

    if_block: "if" expression "do" code_block "end"
    variable_assignment: name "=" expression
    print: "print" value | "print" "(" expression ")"
    to_string: "to_string" value | "to_string" "(" expression ")"

    expression: unary_expression | binary_expression | value | "(" expression ")" | to_string

    binary_expression: expression binary_operator expression
    unary_expression: unary_operator expression
    value: variable_value | string_value | number_value | boolean_value
    variable_value: name

    name: NAME

    boolean_value: TRUE_VALUE | FALSE_VALUE
    binary_operator: IS_OP | LT_OP | GT_OP | ADD_OP |SUB_OP | MUL_OP |DIV_OP | AND_OP | OR_OP
    IS_OP: "is"
    LT_OP: "<"
    GT_OP: ">"
    ADD_OP: "+"
    SUB_OP: "-"
    MUL_OP: "*"
    DIV_OP: "/"
    AND_OP: "and"
    OR_OP: "or"
    NEG_OP: "!"
    TRUE_VALUE: "true"
    FALSE_VALUE: "false"
    
    string_value: STRING 
    number_value: DEC_NUMBER
    unary_operator: NEG_OP

    %import python (NAME, STRING, DEC_NUMBER)
    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
"""


class Parser:
    def __init__(self, grammar):
        self.parser = Lark(grammar, parser="lalr")
        self.transformer = ast_utils.create_transformer(this_module, ToAst())

    def parse(self, code, debug=False):
        cst = self.parser.parse(code)
        if debug:
            print(code)
            print(cst.pretty())
            print(cst)
        return self.transformer.transform(cst)


class Interpreter:
    def __init__(self):
        self.variable_states = defaultdict(lambda: None)
        self.visitors = {
            CodeBlock: self.visit_codeblock,
            IfBlock: self.visit_if_block,
            Expression: self.visit_expression,
            BinaryExpression: self.visit_binary_expression,
            Value: self.visit_value,
            BooleanValue: self.visit_boolean_value,
            StringValue: self.visit_string_value,
            NumberValue: self.visit_number_value,
            VariableValue: self.visit_variable_value,
            Print: self.visit_print,
            ToString: self.visit_to_string,
            VariableAssignment: self.visit_variable_assignment,
            Name: self.visit_name,
            BinaryOperator: self.visit_binary_operator
        }

        self.operator_lookup_table = {
            "is": "==",
            ">": ">",
            "<": "<",
            "and": "&&",
            "or": "||",
            "+": "+",
            "-": "-",
            "*": "*",
            "/": "/"
        }

    def execute(self, ast):
        self.visit(ast)

    def visit_codeblock(self, node):
        for statement in node.statements:
            self.visit(statement)

    def visit_name(self, node):
        return str(node.name)

    def visit_to_string(self, node):
        return f"'{self.visit(node.expression)}'"

    def visit_variable_assignment(self, node):
        self.variable_states[self.visit(node.name)] = self.visit(node.value)
        return

    def visit_if_block(self, node):
        conditional = self.visit(node.cond)
        if conditional:
            return self.visit(node.then)

    def visit_expression(self, node):
        return self.visit(node.value)

    def visit_value(self, node):
        return self.visit(node.value)

    def visit_boolean_value(self, node):
        return node.value == "true"

    def visit_string_value(self, node):
        return f"'{node.value}'"

    def visit_number_value(self, node):
        return float(node.value)

    def visit_variable_value(self, node):
        return self.variable_states[self.visit(node.value)]


    def visit_binary_expression(self, node):
        left_expression = self.visit(node.left_expression)
        right_expression = self.visit(node.right_expression)
        operator = self.visit(node.operator)

        return eval(f"{left_expression} {operator} {right_expression}")

    def visit_binary_operator(self, node):
        return self.operator_lookup_table[str(node.value)]

    def visit_print(self, node):
        print(self.visit(node.value))

    def visit(self, node):
        visitor = self.visitors[type(node)]
        return visitor(node)


code_example1 = """
    a = 1
    print a
    a = 3 * (2 + 1) * 2
    print(a + 1)
"""

code_example2 = """
    a = 1
    if (a < 2) do
      print "done lena asdfasdf"
    end
"""


code_example3 = """
    b = 2
    a = 3 * b
    sum = "sum is " + to_string(b + a)
    print sum
    print "foo"
"""

ast = Parser(grammar).parse(code_example3, debug=True)
print(ast)

interpreter = Interpreter()
interpreter.execute(ast)

print(interpreter.variable_states)
