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
            log_with_data(logger, logging.INFO, "Starting equation solve", input_data)
            
            # parse
            sympy_expr = expr.to_sympy()[0]
            log_with_data(logger, logging.INFO, "Parsed to SymPy", {
                'sympy_expr': str(sympy_expr),
                'expr_type': str(type(sympy_expr))
            })
            
            # tool result
            tool_result = ToolResult.from_tool(
                name="solve_equation",
                input_data=input_data,
                output=""
            )

            # extract values if list of equations
            if isinstance(sympy_expr, list):
                solutions = [eq.rhs for eq in sympy_expr if isinstance(eq, sp.Eq)]
                log_with_data(logger, logging.INFO, "Extracted solutions", {
                    'solutions': [str(s) for s in solutions]
                })
            else:
                # otherwise solve
                solutions = solve(sympy_expr, var_name)
                log_with_data(logger, logging.INFO, "Solve result", {
                    'solutions': [str(s) for s in solutions] if solutions else []
                })

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
                tool_result.output = result

                log_with_data(logger, logging.INFO, "Final solution", {
                    'result': result,
                    'numeric_values': numeric_sols
                })
                
                # return both formatted result and numeric values
                return f"NUMERIC_VALUES={numeric_sols}|RESULT={result}"
            
            result = "No solution found"
            tool_result.output = result
            
            log_with_data(logger, logging.INFO, "No solutions", {
                'equation': str(sympy_expr)
            })
            
            return result
            
        except Exception as e:
            log_with_data(logger, logging.ERROR, "Error in solve_equation", {
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
            log_with_data(logger, logging.INFO, "performing arithmetic operation", {
                'input_latex': expr.latex
            })
            
            sympy_expr = expr.to_sympy()[0]
            log_with_data(logger, logging.DEBUG, "parsed expression", {
                'sympy_expr': str(sympy_expr)
            })
            
            try:
                result = float(sp.N(sympy_expr))
                log_with_data(logger, logging.INFO, "arithmetic result", {
                    'input': str(sympy_expr),
                    'result': result
                })
                return str(result)
            except TypeError:
                # try evaluating the expression if direct conversion fails
                result = sympy_expr.evalf()
                numeric_result = float(result)
                log_with_data(logger, logging.INFO, "arithmetic result after evaluation", {
                    'input': str(sympy_expr),
                    'result': numeric_result
                })
                return str(numeric_result)
                
        except Exception as e:
            log_with_data(logger, logging.ERROR, "error in arithmetic operation", {
                'input_expr': str(expr.latex),
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            raise

    agent.tool(retrieve_similar)