from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai._result import ToolRetryError
from models import MathDependencies, PartialSolution, Solution
from dotenv import load_dotenv
import os
from logger import setup_logger, log_with_data
import logging
from tools import register_tools
from prompts import MATH_SOLVER_PROMPT
from retriever import MathRetriever

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

logger = setup_logger('solver')

class MathSolver:
    def __init__(self):
        logger.info("Initializing MathSolver")
        self.retriever = MathRetriever()
        self.agent = Agent(
            'openai:gpt-4',
            deps_type=MathDependencies,
            result_type=PartialSolution,
            system_prompt=MATH_SOLVER_PROMPT
        )
        
        register_tools(self.agent)

    async def solve_problem(self, problem: str) -> Solution:
        try:
            # retrieve similar examples
            similar_examples = self.retriever.get_similar_examples(problem)
            
            # add to context
            examples_context = "\n\n".join([
                f"Similar problem:\n{ex.problem}\nSolution approach:\n{ex.solution}"
                for ex in similar_examples
            ])
            
            context = f"""
            Here are some similar problems and their solutions for reference:
            
            {examples_context}
            
            Now solve this problem:
            {problem}
            """
            
            log_with_data(logger, logging.INFO, "Starting problem solution", {
                'problem': problem
            })
            
            deps = MathDependencies(original_problem=problem)
            max_steps = 10
            max_retries = 3
            
            while deps.current_step < max_steps:
                try:
                    log_with_data(logger, logging.INFO, f"Processing step {deps.current_step + 1}", {
                        'step_number': deps.current_step + 1,
                        'previous_steps': [
                            {
                                'step': i,
                                'reasoning': step.reasoning,
                                'method': step.method,
                                'result': step.result
                            }
                            for i, step in enumerate(deps.previous_steps)
                        ]
                    })
                    
                    retry_count = 0
                    while retry_count < max_retries:
                        try:
                            logger.debug("Calling agent.run")
                            step_result = await self.agent.run(
                                context,
                                deps=deps
                            )
                            
                            log_with_data(logger, logging.INFO, "Step result", {
                                'step_number': deps.current_step + 1,
                                'reasoning': step_result.data.reasoning,
                                'method': str(step_result.data.method),
                                'result': step_result.data.result,
                                'numeric_values': step_result.data.numeric_values
                            })
                            
                            # add step to ctx
                            deps.previous_steps.append(step_result.data)
                            deps.current_step += 1
                            
                            print(f"\nStep {deps.current_step}:")
                            print(f"Reasoning: {step_result.data.reasoning}")
                            print(f"Method: {step_result.data.method}")
                            print(f"Expression: {step_result.data.expression}")
                            
                            # check the final solution flag
                            if step_result.data.is_final:
                                print("\nSolution found!")
                                return Solution.from_partial_solutions(deps.previous_steps)
                            
                            break
                            
                        except (UnexpectedModelBehavior, ToolRetryError) as e:
                            retry_count += 1
                            log_with_data(logger, logging.WARNING, "Retrying step", {
                                'retry_count': retry_count,
                                'reason': str(e)
                            })
                            if retry_count >= max_retries:
                                raise
                    
                except Exception as e:
                    logger.error(f"Error in step {deps.current_step}: {str(e)}", exc_info=True)
                    raise
            
            print("\nMaximum number of steps reached!")
            return Solution.from_partial_solutions(deps.previous_steps)
            
        except Exception as e:
            logger.error(f"Error solving problem: {str(e)}", exc_info=True)
            raise