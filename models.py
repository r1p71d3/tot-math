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
    
    # TODO: make more robust
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
            
            # substitute values
            formatted_latex = self.latex
            for symbol in self.symbols:
                if symbol.value is not None:
                    value_str = str(symbol.value)
                    name = symbol.name
                    
                    if formatted_latex == name:
                        formatted_latex = value_str
                        continue
                        
                    # handle various symbol contexts
                    formatted_latex = formatted_latex.replace(f" {name} ", f" {value_str} ")
                    if formatted_latex.startswith(f"{name} "):
                        formatted_latex = f"{value_str} " + formatted_latex[len(name)+1:]
                    if formatted_latex.endswith(f" {name}"):
                        formatted_latex = formatted_latex[:-len(name)-1] + f" {value_str}"
                    
                    # handle special characters
                    formatted_latex = formatted_latex.replace(f"{name}\\", f"{value_str}\\")
                    formatted_latex = formatted_latex.replace(f"{name}^", f"{value_str}^")
                    formatted_latex = formatted_latex.replace(f"{name}+", f"{value_str}+")
                    formatted_latex = formatted_latex.replace(f"{name}-", f"{value_str}-")
                    formatted_latex = formatted_latex.replace(f"{name}*", f"{value_str}*")
                    formatted_latex = formatted_latex.replace(f"{name}/", f"{value_str}/")
                    formatted_latex = formatted_latex.replace(name + "}", value_str + "}")
            
            try:
                sympy_expr = latex2sympy(formatted_latex)
            except Exception:
                # handle common arithmetic expressions if latex2sympy fails
                if '\\frac' in formatted_latex and '\\cdot' in formatted_latex:
                    parts = formatted_latex.split('\\frac{')[1].split('}{')
                    numerator = parts[0].replace('\\cdot', '*')
                    denominator = parts[1].split('}')[0]
                    rest = formatted_latex.split('}')[-1].strip()
                    
                    if rest:
                        rest = rest.replace('\\cdot', '*')
                        simple_expr = f"({numerator})/({denominator}){rest}"
                    else:
                        simple_expr = f"({numerator})/({denominator})"
                    
                    sympy_expr = sp.sympify(simple_expr)
                else:
                    simple_expr = formatted_latex.replace('\\cdot', '*')
                    sympy_expr = sp.sympify(simple_expr)
            
            log_with_data(logger, logging.DEBUG, "conversion result", {
                'input_latex': formatted_latex,
                'output_sympy': str(sympy_expr),
                'sympy_type': str(type(sympy_expr))
            })
            
            return [sympy_expr]
        except Exception as e:
            log_with_data(logger, logging.ERROR, "LaTeX conversion error", {
                'input_latex': self.latex,
                'formatted_latex': formatted_latex if 'formatted_latex' in locals() else None,
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            raise

    class Config:
        json_encoders = {
            # JSON encoder for LaTeX strings
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

class ThoughtState(BaseModel):
    """Represents a state in the thought tree"""
    state_id: str
    parent_id: Optional[str] = None
    reasoning: str
    expression: Optional[Expression] = None
    method: Optional[MathMethod] = None
    result: Optional[str] = None
    numeric_values: Optional[Dict[str, float]] = None
    value_estimate: float  # Score from evaluation
    is_final: bool = False
    children: List[str] = Field(default_factory=list)
    steps_taken: List[str] = Field(default_factory=list)  # Track solution path

class ThoughtTree(BaseModel):
    """Represents the tree of thoughts"""
    states: Dict[str, ThoughtState] = Field(default_factory=dict)
    root_id: str
    current_id: str