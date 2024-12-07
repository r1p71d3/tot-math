import streamlit as st
import asyncio
from solver import MathSolver
import sys
from io import StringIO
import contextlib

@contextlib.contextmanager
def capture_output():
    """Capture stdout to a string"""
    old_stdout = sys.stdout
    stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old_stdout

async def solve_math_problem(problem: str):
    """Solve a math problem and return both the solution and captured output"""
    solver = MathSolver()
    
    with capture_output() as captured:
        solution = await solver.solve_problem(problem)
    
    return solution, captured.getvalue()

st.title('Math Problem Solver')

st.markdown("""
This app solves mathematical problems step by step. Enter your problem below and click 'Solve'.
""")

problem = st.text_area("Enter your math problem:", height=100)

if st.button('Solve'):
    if problem:
        with st.spinner('Solving...'):
            solution, thought_process = asyncio.run(solve_math_problem(problem))
            
            st.success("Solution Found!")
            
            if solution.numeric_values:
                st.write("Numeric Answer:", solution.numeric_values)
            if solution.symbolic_result:
                st.write("Result:", solution.symbolic_result)
            st.write("Explanation:", solution.explanation)
            
            with st.expander("See solution steps"):
                st.text(thought_process)
    else:
        st.error("Please enter a problem first!")

st.markdown("""
---
This solver can handle various types of math problems including:
- Equations
- Arithmetic
- Factoring
- Integration
- Differentiation
- And more!
""") 