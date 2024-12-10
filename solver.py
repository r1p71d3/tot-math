from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai._result import ToolRetryError
from models import MathDependencies, PartialSolution, Solution, ThoughtState, ThoughtTree
from retriever import MathExample, MathRetriever
from dotenv import load_dotenv
import os
from logger import setup_logger, log_with_data
import logging
from tools import register_tools
from prompts import MATH_SOLVER_PROMPT, THOUGHT_GENERATION_PROMPT, THOUGHT_EVALUATION_PROMPT, SOLUTION_EVALUATION_PROMPT
from typing import List, Tuple

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

    async def evaluate_thought(self, state: ThoughtState, context: str) -> float:
        """Evaluate a thought state based on its potential"""
        try:
            if state.is_final:
                # Evaluate complete solution
                response = await self.agent.run(
                    SOLUTION_EVALUATION_PROMPT.format(
                        problem=context,
                        reasoning=state.reasoning,
                        result=state.result
                    ),
                    deps=MathDependencies(original_problem=context)
                )
                if response.data and response.data.numeric_values:
                    correctness = response.data.numeric_values.get('correctness', 0)
                    completeness = response.data.numeric_values.get('completeness', 0)
                    return (correctness + completeness) / 2
            else:
                # Evaluate intermediate step
                response = await self.agent.run(
                    THOUGHT_EVALUATION_PROMPT.format(
                        problem=context,
                        reasoning=state.reasoning
                    ),
                    deps=MathDependencies(original_problem=context)
                )
                if response.data and response.data.numeric_values:
                    return response.data.numeric_values.get('score', 0)
            return 0.0
        except Exception as e:
            logger.error(f"Error evaluating thought: {str(e)}")
            return 0.0

    async def generate_next_thoughts(self, state: ThoughtState, context: str, examples: List[MathExample], k: int = 3) -> List[ThoughtState]:
        """Generate k possible next thoughts"""
        try:
            examples_text = "\n".join([
                f"Example {i+1}:\nProblem: {ex.problem}\nSolution: {ex.solution}"
                for i, ex in enumerate(examples)
            ])
            
            response = await self.agent.run(
                THOUGHT_GENERATION_PROMPT.format(
                    problem=context,
                    reasoning=state.reasoning,
                    examples_text=examples_text,
                    k=k
                ),
                deps=MathDependencies(original_problem=context)
            )
            
            thoughts = []
            if response.data:
                thought = ThoughtState(
                    state_id=f"{state.state_id}.{len(thoughts)}",
                    parent_id=state.state_id,
                    reasoning=response.data.reasoning,
                    expression=response.data.expression,
                    method=response.data.method,
                    result=response.data.result,
                    value_estimate=0.0,
                    is_final=response.data.is_final,
                    steps_taken=state.steps_taken + [response.data.reasoning]
                )
                thoughts.append(thought)
            
            return thoughts
        except Exception as e:
            logger.error(f"Error generating thoughts: {str(e)}")
            return []

    async def get_solution_path(self, tree: ThoughtTree, final_state: ThoughtState) -> List[ThoughtState]:
        """Get the path from root to the final state"""
        path = []
        current = final_state
        while current:
            path.append(current)
            if current.parent_id:
                current = tree.states.get(current.parent_id)
            else:
                break
        return list(reversed(path))

    async def solve_problem(self, problem: str) -> Tuple[Solution, ThoughtTree, List[MathExample]]:
        try:
            # Get similar examples
            similar_examples = self.retriever.get_similar_examples(problem)
            logger.info(f"Retrieved {len(similar_examples)} similar examples")
            
            # Initialize tree with root
            root = ThoughtState(
                state_id="0",
                reasoning=f"Initial problem: {problem}",
                value_estimate=1.0,
                is_final=False,
                steps_taken=[]
            )
            
            tree = ThoughtTree(states={root.state_id: root}, root_id="0", current_id="0")
            
            # Beam search parameters
            max_steps = 10
            beam_width = 3
            
            # Beam search through thought tree
            for step in range(max_steps):
                logger.info(f"Processing step {step + 1}")
                
                # Get current beam of states
                current_states = sorted(
                    tree.states.values(),
                    key=lambda x: x.value_estimate,
                    reverse=True
                )[:beam_width]
                
                # Generate and evaluate next thoughts
                all_next_thoughts = []
                for state in current_states:
                    # Generate next possible steps
                    next_thoughts = await self.generate_next_thoughts(
                        state, problem, similar_examples
                    )
                    logger.info(f"Generated {len(next_thoughts)} thoughts from state {state.state_id}")
                    
                    # Evaluate each thought
                    for thought in next_thoughts:
                        thought.value_estimate = await self.evaluate_thought(thought, problem)
                        logger.info(f"Thought {thought.state_id} evaluated with score {thought.value_estimate}")
                    
                    all_next_thoughts.extend(next_thoughts)
                
                if not all_next_thoughts:
                    logger.info("No more thoughts generated, ending search")
                    break
                
                # Update tree with new thoughts
                for thought in all_next_thoughts:
                    tree.states[thought.state_id] = thought
                    tree.states[thought.parent_id].children.append(thought.state_id)
                
                # Check for solution
                best_thought = max(all_next_thoughts, key=lambda x: x.value_estimate)
                if best_thought.is_final and best_thought.value_estimate > 0.95:
                    logger.info(f"Found solution with score {best_thought.value_estimate}")
                    return self._create_solution(best_thought), tree, similar_examples
            
            # Return best attempt if no complete solution found
            best_state = max(tree.states.values(), key=lambda x: x.value_estimate)
            logger.info(f"Returning best attempt with score {best_state.value_estimate}")
            return self._create_solution(best_state), tree, similar_examples
            
        except Exception as e:
            logger.error(f"Error solving problem: {str(e)}")
            raise

    def _create_solution(self, state: ThoughtState) -> Solution:
        """Create a Solution object from a ThoughtState"""
        return Solution(
            explanation=state.reasoning,
            numeric_values=state.numeric_values,
            symbolic_result=state.result
        )