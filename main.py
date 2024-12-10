from solver import MathSolver

async def main():
    solver = MathSolver()
    
    # examples
    problems = [
        # "Solve the equation x^2 + 2x + 1 = 0",
        # "Solve the equation x^2 - 1432.0012x + 1.7184 = 0",
        # "Jack had 10 books. He gave 5 books to Alice and 3 books to Bob. He then bought 4 more books. How many books does Jack have now?",
        # "Simplify the expression (x + 1)(x - 1)",
        # "A factory uses 83874.24 liters of raw material every 5 days. 74% of the material each day is used to produce item A, 13.5% is discarded, and the rest is used for item B. If the factory operates 24/7, what is the square root of the amount of material in liters used to produce product B per hour in liters?",
        "How many vertical asymptotes does the graph of $y=\\frac{x-2}{x^2+x-6}$ have?",
        # "Tom has a red marble, a green marble, a blue marble, and 3 identical yellow marbles. How many distinct groups of 2 marbles can Tom choose?",
    ]
    
    for problem in problems:
        print(f"\nSolving problem: {problem}")
        solution = await solver.solve_problem(problem)
        print("\nFinal Solution:")
        print(f"Explanation: {solution.explanation}")
        if solution.numeric_values:
            print(f"Numeric values: {solution.numeric_values}")
        if solution.symbolic_result:
            print(f"Symbolic result: {solution.symbolic_result}")
        print("-" * 50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
