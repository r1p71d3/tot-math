from pydantic_ai import RunContext, Agent, ModelRetry
from sympy import solve, factor, expand, integrate, diff, Eq
import sympy as sp
from models import Expression, MathDependencies, PartialSolution, ToolResult
from logger import setup_logger, log_with_data
import logging
from retriever import retrieve_similar

logger = setup_logger('tools')

def register_tools(agent):
    @agent.tool
    def solve_equation(ctx: RunContext[MathDependencies], expr: Expression, var_name: str) -> str:
        """Solves an equation for a given variable."""
        try:
            input_data = {
                'variable': var_name,
                'latex_expr': expr.latex
            }
            log_with_data(logger, logging.INFO, "[SOLVE] Starting equation solve", input_data)
            
            # convert to sympy
            if '=' in expr.latex:
                left_str, right_str = expr.latex.split('=')
                
                left_expr = Expression(
                    latex=left_str.strip(),
                    symbols=expr.symbols
                )
                right_expr = Expression(
                    latex=right_str.strip(),
                    symbols=expr.symbols
                )
                
                left_sympy = left_expr.to_sympy()[0]
                right_sympy = right_expr.to_sympy()[0]
                
                equation = sp.Eq(left_sympy, right_sympy)
                
                log_with_data(logger, logging.INFO, "[SOLVE] Created equation", {
                    'left_side': str(left_sympy),
                    'right_side': str(right_sympy),
                    'equation': str(equation)
                })
            else:
                # if no equals sign, assume equation = 0 (default sympy behavior)
                sympy_expr = expr.to_sympy()[0]
                equation = sp.Eq(sympy_expr, 0)
                
                log_with_data(logger, logging.INFO, "[SOLVE] Created equation from expression", {
                    'expression': str(sympy_expr),
                    'equation': str(equation)
                })
            
            # solve
            solutions = sp.solve(equation, var_name)
            
            log_with_data(logger, logging.INFO, "[SOLVE] Found solutions", {
                'solutions': [str(s) for s in solutions]
            })
            
            # format
            if solutions:
                solution_strs = []
                numeric_sols = []
                
                for sol in solutions:
                    try:
                        val = float(sol.evalf())
                        numeric_sols.append(val)
                        solution_strs.append(f"{var_name} = {val:.10f}")
                    except:
                        solution_strs.append(f"{var_name} = {sol}")
                
                result = ", ".join(solution_strs)
                
                log_with_data(logger, logging.INFO, "[SOLVE] Final solution", {
                    'result': result,
                    'numeric_values': numeric_sols
                })
                
                return f"NUMERIC_VALUES={numeric_sols}|RESULT={result}"
            
            result = "No solution found"
            log_with_data(logger, logging.INFO, "[SOLVE] No solutions", {
                'equation': str(equation)
            })
            
            return result
            
        except Exception as e:
            log_with_data(logger, logging.ERROR, "[SOLVE] Error in solve_equation", {
                'input_expr': str(expr.latex),
                'variable': var_name,
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            raise

    @agent.tool
    def factor_expression(ctx: RunContext[MathDependencies], expr: Expression) -> str:
        """Factors a mathematical expression."""
        try:
            logger.info("Factoring expression")
            sympy_expr = expr.to_sympy()[0]
            
            if isinstance(sympy_expr, sp.Eq):
                lhs = factor(sympy_expr.lhs)
                rhs = factor(sympy_expr.rhs)
                result = sp.Eq(lhs, rhs)
            else:
                result = factor(sympy_expr)
            
            result_str = str(result)
            logger.info(f"Factor result: {result_str}")
            
            # formatting
            tool_result = ToolResult(
                tool_name="factor_expression",
                input_data={
                    'expression': expr.latex,
                    'symbols': [
                        {'name': s.name, 'assumptions': s.assumptions}
                        for s in expr.symbols
                    ]
                },
                output=result_str
            )
            
            # store in step
            if hasattr(ctx.deps, 'previous_steps') and ctx.deps.previous_steps:
                ctx.deps.previous_steps[-1].tool_results.append(tool_result)
            
            return result_str
            
        except Exception as e:
            logger.error(f"Error in factor_expression: {str(e)}", exc_info=True)
            raise

    @agent.tool
    def expand_expression(ctx: RunContext[MathDependencies], expr: Expression) -> str:
        """Expands a mathematical expression."""
        sympy_expr = expr.to_sympy()[0]
        if isinstance(sympy_expr, sp.Eq):
            # expand both sides if equation
            lhs = expand(sympy_expr.lhs)
            rhs = expand(sympy_expr.rhs)
            result = sp.Eq(lhs, rhs)
        else:
            result = expand(sympy_expr)
        return str(result)

    @agent.tool
    def integrate_expression(ctx: RunContext[MathDependencies], expr: Expression, var_name: str) -> str:
        """Integrates an expression with respect to a variable."""
        sympy_expr = expr.to_sympy()[0]
        if isinstance(sympy_expr, sp.Eq):
            sympy_expr = sympy_expr.lhs - sympy_expr.rhs
        var = sp.Symbol(var_name)
        result = integrate(sympy_expr, var)
        return str(result)

    @agent.tool
    def differentiate_expression(ctx: RunContext[MathDependencies], expr: Expression, var_name: str) -> str:
        """Differentiates an expression with respect to a variable."""
        sympy_expr = expr.to_sympy()[0]
        if isinstance(sympy_expr, sp.Eq):
            sympy_expr = sympy_expr.lhs - sympy_expr.rhs
        var = sp.Symbol(var_name)
        result = diff(sympy_expr, var)
        return str(result)

    @agent.tool
    def arithmetic_operation(ctx: RunContext[MathDependencies], expr: Expression) -> str:
        """Performs basic arithmetic operations."""
        try:
            log_with_data(logger, logging.INFO, "[CALC] Starting arithmetic calculation", {
                'input_latex': expr.latex
            })
            
            sympy_expr = expr.to_sympy()[0]
            log_with_data(logger, logging.DEBUG, "[CALC] Parsed expression", {
                'sympy_expr': str(sympy_expr)
            })
            
            try:
                # sub vals, simplify
                if isinstance(sympy_expr, sp.Basic):
                    subs_expr = sympy_expr
                    for symbol in expr.symbols:
                        if symbol.value is not None:
                            subs_expr = subs_expr.subs(sp.Symbol(symbol.name), symbol.value)
                    
                    result = float(subs_expr.evalf())
                else:
                    result = float(sympy_expr)
                    
                log_with_data(logger, logging.INFO, "[CALC] Calculation result", {
                    'input': str(sympy_expr),
                    'substituted': str(subs_expr) if 'subs_expr' in locals() else str(sympy_expr),
                    'result': result
                })
                return str(result)
                
            except (TypeError, ValueError) as e:
                log_with_data(logger, logging.WARNING, "[CALC] Direct evaluation failed, trying alternative method", {
                    'error': str(e)
                })
                # if direct conversion fails, try eval step by step
                try:
                    # sub all vals
                    subs_expr = sympy_expr
                    for symbol in expr.symbols:
                        if symbol.value is not None:
                            subs_expr = subs_expr.subs(sp.Symbol(symbol.name), symbol.value)
                    
                    # simplify, eval
                    simplified = sp.simplify(subs_expr)
                    evaluated = simplified.doit()
                    result = float(evaluated.evalf())
                    
                    log_with_data(logger, logging.INFO, "[CALC] Calculation result after step-by-step evaluation", {
                        'input': str(sympy_expr),
                        'substituted': str(subs_expr),
                        'simplified': str(simplified),
                        'evaluated': str(evaluated),
                        'result': result
                    })
                    return str(result)
                except Exception as inner_e:
                    raise Exception(f"Failed to evaluate expression after substitution: {str(inner_e)}")
                
        except Exception as e:
            log_with_data(logger, logging.ERROR, "[CALC] Error in arithmetic operation", {
                'input_expr': str(expr.latex),
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            raise

    agent.tool(retrieve_similar)