from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from sympy import Symbol, Expr, Eq, N
import sympy as sp
from enum import Enum
from latex2sympy2 import latex2sympy
from logger import setup_logger, log_with_data
import logging

logger = setup_logger('models')

class MathMethod(str, Enum):
    SOLVE = "solve"
    FACTOR = "factor"
    EXPAND = "expand"
    SIMPLIFY = "simplify"
    INTEGRATE = "integrate"
    DIFFERENTIATE = "differentiate"
    ARITHMETIC = "arithmetic"
    EVALUATE = "evaluate"

class MathSymbol(BaseModel):
    name: str
    assumptions: Dict[str, bool] = Field(default_factory=dict)
    value: Optional[Union[int, float]] = None

    def to_sympy(self) -> Symbol:
        return sp.Symbol(self.name, **self.assumptions)

class Expression(BaseModel):
    symbols: List[MathSymbol]
    latex: str
    
    def to_sympy(self) -> List[Union[Expr, Eq]]:
        """converts LaTeX expression to SymPy expression"""
        try:
            log_with_data(logger, logging.INFO, "converting LaTeX to SymPy", {
                'input_latex': self.latex,
                'symbols': [
                    {'name': s.name, 'assumptions': s.assumptions, 'value': s.value}
                    for s in self.symbols
                ]
            })
            
            # clean up latex
            formatted_latex = self.latex
            formatted_latex = formatted_latex.replace('\\_', '_')
            formatted_latex = formatted_latex.replace('{{', '{').replace('}}', '}')  # handle double braces
            
            # substitute values using regex
            import re
            for symbol in self.symbols:
                if symbol.value is not None:
                    value_str = str(symbol.value)
                    name = symbol.name.replace('_', '')
                    
                    if formatted_latex == name:
                        formatted_latex = value_str
                        continue
                    
                    formatted_latex = re.sub(
                        fr'\b{re.escape(name)}\b',
                        value_str,
                        formatted_latex
                    )
            
            try:
                if any(op in formatted_latex for op in ['*', '+', '-', '/', '(', ')']) or '\\frac' in formatted_latex or '\\sqrt' in formatted_latex:
                    expr_str = formatted_latex
                    
                    # handle fractions
                    if '\\frac' in expr_str:
                        while '\\frac' in expr_str:
                            # num
                            frac_start = expr_str.find('\\frac')
                            brace_count = 0
                            num_start = expr_str.find('{', frac_start) + 1
                            num_end = num_start
                            for i in range(num_start, len(expr_str)):
                                if expr_str[i] == '{':
                                    brace_count += 1
                                elif expr_str[i] == '}':
                                    if brace_count == 0:
                                        num_end = i
                                        break
                                    brace_count -= 1
                            
                            # denom
                            brace_count = 0
                            denom_start = expr_str.find('{', num_end) + 1
                            denom_end = denom_start
                            for i in range(denom_start, len(expr_str)):
                                if expr_str[i] == '{':
                                    brace_count += 1
                                elif expr_str[i] == '}':
                                    if brace_count == 0:
                                        denom_end = i
                                        break
                                    brace_count -= 1
                            
                            # extract both
                            numerator = expr_str[num_start:num_end]
                            denominator = expr_str[denom_start:denom_end]
                            
                            before = expr_str[:frac_start]
                            after = expr_str[denom_end + 1:]
                            expr_str = f"{before}(({numerator})/({denominator})){after}"
                    
                    # handle square roots
                    if '\\sqrt' in expr_str:
                        while '\\sqrt' in expr_str:
                            sqrt_start = expr_str.find('\\sqrt')
                            brace_count = 0
                            content_start = expr_str.find('{', sqrt_start) + 1
                            content_end = content_start
                            for i in range(content_start, len(expr_str)):
                                if expr_str[i] == '{':
                                    brace_count += 1
                                elif expr_str[i] == '}':
                                    if brace_count == 0:
                                        content_end = i
                                        break
                                    brace_count -= 1
                            
                            # extract sqrt content
                            content = expr_str[content_start:content_end]
                            
                            # substitute values in content first
                            content_with_values = content
                            for symbol in self.symbols:
                                if symbol.value is not None:
                                    content_with_values = re.sub(
                                        fr'\b{re.escape(symbol.name)}\b',
                                        str(symbol.value),
                                        content_with_values
                                    )
                            
                            try:
                                # try evaluating the content numerically
                                numeric_content = float(sp.sympify(content_with_values))
                                result = numeric_content ** 0.5
                                before = expr_str[:sqrt_start]
                                after = expr_str[content_end + 1:]
                                expr_str = f"{before}{result}{after}"
                            except (ValueError, TypeError, sp.SympifyError):
                                # if numeric eval fails, use sympy's power
                                before = expr_str[:sqrt_start]
                                after = expr_str[content_end + 1:]
                                expr_str = f"{before}(({content_with_values})**(1/2)){after}"
                    
                    # convert operators
                    expr_str = expr_str.replace('\\cdot', '*')
                    expr_str = expr_str.replace('\\times', '*')
                    expr_str = expr_str.replace('\\div', '/')
                    
                    log_with_data(logger, logging.DEBUG, "converted expression", {
                        'original': formatted_latex,
                        'converted': expr_str
                    })
                    
                    # always use sympy's sympify
                    sympy_expr = sp.sympify(expr_str)
                    
                    log_with_data(logger, logging.DEBUG, "parsed to sympy", {
                        'input': formatted_latex,
                        'parsed': str(sympy_expr),
                        'type': str(type(sympy_expr))
                    })
                    
                    return [sympy_expr]
                    
                else:
                    # use latex2sympy for pure latex
                    sympy_expr = latex2sympy(formatted_latex)
                    return [sympy_expr]
                    
            except Exception as e:
                log_with_data(logger, logging.ERROR, "latex conversion error", {
                    'input_latex': self.latex,
                    'formatted_latex': formatted_latex,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                })
                raise
            
        except Exception as e:
            log_with_data(logger, logging.ERROR, "latex conversion error", {
                'input_latex': self.latex,
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            raise

    class Config:
        json_encoders = {
            str: lambda v: v.replace('\\', '\\\\') if '\\' in v else v
        }

class ValidationResult(BaseModel):
    is_valid: bool
    reasoning: str

class ToolResult(BaseModel):
    tool_name: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output: str

    @classmethod
    def from_tool(cls, name: str, input_data: Optional[Dict[str, Any]] = None, output: str = "") -> 'ToolResult':
        return cls(
            tool_name=name,
            input_data=input_data or {},
            output=output
        )

class PartialSolution(BaseModel):
    reasoning: str
    expression: Expression
    method: MathMethod
    result: Optional[str] = None
    numeric_values: Optional[Dict[str, float]] = None
    is_final: bool = False
    tool_results: List[ToolResult] = Field(default_factory=list)

class Solution(BaseModel):
    """Final solution model that OpenAI must conform to"""
    explanation: str
    numeric_values: Optional[Dict[str, float]] = None
    symbolic_result: Optional[str] = None
    is_final: bool = True

class MathDependencies(BaseModel):
    current_step: int = 0
    previous_steps: List[PartialSolution] = Field(default_factory=list)
    original_problem: str = ""
    retriever: Optional[Any] = None

class ThoughtState(BaseModel):
    """Represents a state in the thought tree"""
    state_id: str
    parent_id: Optional[str] = None
    reasoning: str
    expression: Optional[Expression] = None
    method: Optional[MathMethod] = None
    result: Optional[str] = None
    numeric_values: Optional[Dict[str, float]] = None
    value_estimate: float  # eval score
    is_final: bool = False
    children: List[str] = Field(default_factory=list)
    steps_taken: List[str] = Field(default_factory=list)  # track path

class ThoughtTree(BaseModel):
    """Represents the tree of thoughts"""
    states: Dict[str, ThoughtState] = Field(default_factory=dict)
    root_id: str
    current_id: str