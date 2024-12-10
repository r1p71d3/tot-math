import streamlit as st
import asyncio
from solver import MathSolver
import sys
from io import StringIO
import contextlib
import networkx as nx
import matplotlib.pyplot as plt

@contextlib.contextmanager
def capture_output():
    """capture stdout to a string"""
    old_stdout = sys.stdout
    stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old_stdout

def visualize_thought_tree(tree):
    """create a visualization of the thought tree"""
    G = nx.DiGraph()
    
    for state_id, state in tree.states.items():
        G.add_node(state_id, 
                  value=f"{state.value_estimate:.2f}",
                  reasoning=state.reasoning[:50] + "...")
        if state.parent_id:
            G.add_edge(state.parent_id, state_id)
    
    pos = nx.spring_layout(G)
    
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=2000, font_size=8, arrows=True)
    
    labels = nx.get_node_attributes(G, 'value')
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    return plt

async def solve_math_problem(problem: str):
    """solve a math problem and return solution, thought process, and visualization"""
    solver = MathSolver()
    
    with capture_output() as captured:
        solution, tree, examples = await solver.solve_problem(problem)
        path = await solver.get_solution_path(tree, max(tree.states.values(), key=lambda x: x.value_estimate))
    
    return solution, captured.getvalue(), tree, examples, path

st.title('Math Problem Solver (Tree of Thoughts)')

st.markdown("""
This solver uses the Tree of Thoughts method to explore multiple solution paths simultaneously.
Enter your problem below and click 'Solve' to see the solution process.
""")

problem = st.text_area("Enter your math problem:", height=100)

if st.button('Solve'):
    if problem:
        with st.spinner('Solving...'):
            solution, thought_process, tree, examples, path = asyncio.run(solve_math_problem(problem))
            
            st.subheader("Similar Examples Used:")
            for i, example in enumerate(examples, 1):
                with st.expander(f"Example {i} - Similarity: {example.similarity:.2f}"):
                    st.write("Problem:", example.problem)
                    st.write("Solution:", example.solution)
            
            st.success("Solution Found!")
            
            if solution.numeric_values:
                st.write("Numeric Answer:", solution.numeric_values)
            if solution.symbolic_result:
                st.write("Result:", solution.symbolic_result)
            st.write("Explanation:", solution.explanation)
            
            st.subheader("Solution Process:")
            for i, state in enumerate(path):    # TODO: fix this
                with st.expander(f"Step {i+1} - Score: {state.value_estimate:.2f}"):
                    st.write("Reasoning:", state.reasoning)
                    if state.result:
                        st.write("Result:", state.result)
            
            st.subheader("Full Search Tree")
            fig = visualize_thought_tree(tree)
            st.pyplot(fig)
            
            # TODO: fix this
            with st.expander("See raw solution steps"):
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

The Tree of Thoughts approach:
1. Generates multiple possible solution paths
2. Evaluates the promise of each path
3. Explores the most promising paths further
4. Finds the best solution among all attempts
""") 