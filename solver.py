from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai._result import ToolRetryError
from models import MathDependencies, PartialSolution, Solution, ThoughtState, ThoughtTree
from retriever import MathExample, MathRetriever
from dotenv import load_dotenv
import os
from logger import setup_logger, log_with_data, LogColors
import logging
from tools import register_tools
from prompts import MATH_SOLVER_PROMPT, THOUGHT_GENERATION_PROMPT, THOUGHT_EVALUATION_PROMPT, SOLUTION_EVALUATION_PROMPT
from typing import List, Tuple

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

logger = setup_logger('solver')

max_steps = 10
beam_width = 3

class MathSolver:
    def __init__(self):
        logger.info(f"{LogColors.HEADER}>>> Initializing Math Problem Solver <<<{LogColors.ENDC}")
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
            # check if valid step and give base score
            if state.reasoning and state.method:
                base_score = 0.5
                
                # + score if result
                if state.result:
                    base_score += 0.2
                
                # + score if numeric
                if state.numeric_values and 'result' in state.numeric_values:
                    base_score += 0.1
                    logger.info(f"{LogColors.OKGREEN}[EVAL] Found numeric result: {state.numeric_values['result']}{LogColors.ENDC}")
                
                # more eval if flagged as final
                if state.is_final:
                    logger.info(f"{LogColors.OKCYAN}[EVAL] Evaluating final solution for state {state.state_id}{LogColors.ENDC}")
                    try:
                        response = await self.agent.run(
                            SOLUTION_EVALUATION_PROMPT.format(
                                problem=context,
                                reasoning=state.reasoning,
                                result=state.result
                            ),
                            deps=self.deps
                        )
                        if response.data and response.data.numeric_values:
                            correctness = response.data.numeric_values.get('correctness', 0)
                            completeness = response.data.numeric_values.get('completeness', 0)
                            score = max(base_score, (correctness + completeness) / 2)
                            logger.info(f"{LogColors.OKGREEN}[SCORE] Final solution: {score:.2f} (correctness: {correctness:.2f}, completeness: {completeness:.2f}){LogColors.ENDC}")
                            return score
                    except Exception as e:
                        logger.warning(f"{LogColors.WARNING}[EVAL] Failed to evaluate final solution: {str(e)}{LogColors.ENDC}")
                        return base_score
                else:
                    # consider reasoning for intermediate steps
                    logger.info(f"{LogColors.OKCYAN}[EVAL] Evaluating intermediate step for state {state.state_id}{LogColors.ENDC}")
                    try:
                        response = await self.agent.run(
                            THOUGHT_EVALUATION_PROMPT.format(
                                problem=context,
                                reasoning=state.reasoning
                            ),
                            deps=self.deps
                        )
                        if response.data and response.data.numeric_values:
                            step_score = response.data.numeric_values.get('score', 0)
                            score = max(base_score, step_score)
                            logger.info(f"{LogColors.OKGREEN}[SCORE] Step evaluation: {score:.2f}{LogColors.ENDC}")
                            return score
                    except Exception as e:
                        logger.warning(f"{LogColors.WARNING}[EVAL] Failed to evaluate step: {str(e)}{LogColors.ENDC}")
                        return base_score
                
                logger.info(f"{LogColors.OKGREEN}[SCORE] Base evaluation: {base_score:.2f}{LogColors.ENDC}")
                return base_score
                
            return 0.0
        except Exception as e:
            logger.error(f"{LogColors.FAIL}[ERROR] Evaluation failed: {str(e)}{LogColors.ENDC}")
            return 0.0

    async def generate_next_thoughts(self, state: ThoughtState, context: str, examples: List[MathExample], k: int = 3) -> List[ThoughtState]:
        """Generate k possible next thoughts"""
        max_attempts = 3  # num of retries
        
        try:
            logger.info(f"{LogColors.OKCYAN}[GENERATE] Starting thought generation for state {state.state_id}{LogColors.ENDC}")
            
            examples_text = "\n".join([
                f"Example {i+1}:\nProblem: {ex.problem}\nSolution: {ex.solution}"
                for i, ex in enumerate(examples)
            ])
            
            logger.info(f"{LogColors.OKBLUE}[PROMPT] Generating thoughts using {len(examples)} example(s){LogColors.ENDC}")
            
            for attempt in range(max_attempts):
                try:
                    if attempt > 0:
                        logger.info(f"{LogColors.WARNING}[RETRY] Attempt {attempt + 1} of {max_attempts}{LogColors.ENDC}")
                    
                    response = await self.agent.run(
                        THOUGHT_GENERATION_PROMPT.format(
                            problem=context,
                            reasoning=state.reasoning,
                            examples_text=examples_text,
                            k=k
                        ),
                        deps=self.deps
                    )
                    
                    thoughts = []
                    if response and response.data:
                        logger.info(f"{LogColors.OKBLUE}[PROCESS] Creating new thought state from response{LogColors.ENDC}")
                        logger.debug(f"Response data: {response.data}")
                        
                        # store numeric
                        numeric_values = {}
                        if hasattr(response.data, 'result'):
                            try:
                                result_value = float(response.data.result)
                                numeric_values['result'] = result_value
                                logger.info(f"{LogColors.OKGREEN}[RESULT] Found numeric result: {result_value}{LogColors.ENDC}")
                            except (ValueError, TypeError):
                                pass
                        
                        thought = ThoughtState(
                            state_id=f"{state.state_id}.{len(thoughts)}",
                            parent_id=state.state_id,
                            reasoning=response.data.reasoning,
                            expression=response.data.expression,
                            method=response.data.method,
                            result=response.data.result,
                            numeric_values=numeric_values,
                            value_estimate=0.0,
                            is_final=response.data.is_final,
                            steps_taken=state.steps_taken + [response.data.reasoning]
                        )
                        thoughts.append(thought)
                        logger.info(f"{LogColors.OKGREEN}[SUCCESS] Created thought with ID {thought.state_id}{LogColors.ENDC}")
                        return thoughts
                    else:
                        logger.warning(f"{LogColors.WARNING}[WARNING] No valid response data received (Attempt {attempt + 1}){LogColors.ENDC}")
                        logger.debug(f"Raw response: {response}")
                        if attempt == max_attempts - 1:
                            logger.error(f"{LogColors.FAIL}[ERROR] Failed to generate valid thoughts after {max_attempts} attempts{LogColors.ENDC}")
                            return []
                        
                except Exception as e:
                    logger.error(f"{LogColors.FAIL}[ERROR] Attempt {attempt + 1} failed: {str(e)}{LogColors.ENDC}")
                    if attempt == max_attempts - 1:
                        raise
                    continue
                
        except Exception as e:
            logger.error(f"{LogColors.FAIL}[ERROR] Failed to generate thoughts: {str(e)}{LogColors.ENDC}")
            logger.exception("Detailed error traceback:")
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
            logger.info(f"{LogColors.HEADER}\n{'='*50}\n>>> Starting to solve problem:\n{problem}\n{'='*50}{LogColors.ENDC}")
            
            # RAG for similar examples
            logger.info(f"{LogColors.OKBLUE}[SEARCH] Finding similar example problems...{LogColors.ENDC}")
            similar_examples = self.retriever.get_similar_examples(problem)
            logger.info(f"{LogColors.OKGREEN}[FOUND] {len(similar_examples)} relevant examples{LogColors.ENDC}")
            
            # create deps
            self.deps = MathDependencies(
                original_problem=problem,
                retriever=self.retriever
            )
            
            # init tree
            logger.info(f"{LogColors.OKBLUE}[INIT] Initializing solution tree{LogColors.ENDC}")
            root = ThoughtState(
                state_id="0",
                reasoning=f"Initial problem: {problem}",
                value_estimate=1.0,
                is_final=False,
                steps_taken=[]
            )
            
            tree = ThoughtTree(states={root.state_id: root}, root_id="0", current_id="0")
            
            # beam search
            for step in range(max_steps):
                logger.info(f"{LogColors.HEADER}\n{'='*30}\n>>> STEP {step + 1}\n{'='*30}{LogColors.ENDC}")
                
                current_states = sorted(
                    tree.states.values(),
                    key=lambda x: x.value_estimate,
                    reverse=True
                )[:beam_width]
                
                logger.info(f"{LogColors.OKBLUE}[EXPLORE] Analyzing {len(current_states)} most promising paths{LogColors.ENDC}")
                
                # gen next thoughts
                all_next_thoughts = []
                for state in current_states:
                    next_thoughts = await self.generate_next_thoughts(
                        state, problem, similar_examples
                    )
                    logger.info(f"{LogColors.OKGREEN}[GENERATE] Created {len(next_thoughts)} new thoughts from state {state.state_id}{LogColors.ENDC}")
                    
                    for thought in next_thoughts:
                        thought.value_estimate = await self.evaluate_thought(thought, problem)
                        logger.info(f"{LogColors.OKCYAN}[SCORE] State {thought.state_id}: {thought.value_estimate:.2f}{LogColors.ENDC}")
                    
                    all_next_thoughts.extend(next_thoughts)
                
                if not all_next_thoughts:
                    logger.info(f"{LogColors.WARNING}[WARNING] No more thoughts generated, ending search{LogColors.ENDC}")
                    break
                
                # check for excellent solution
                current_best = max(tree.states.values(), key=lambda x: x.value_estimate)
                if current_best.is_final and current_best.value_estimate >= 0.98:
                    logger.info(f"{LogColors.OKGREEN}[SUCCESS] Found excellent solution with score {current_best.value_estimate:.2f}, stopping early{LogColors.ENDC}")
                    return self._create_solution(current_best), tree, similar_examples
                
                # check for solution
                best_thought = max(all_next_thoughts, key=lambda x: x.value_estimate)
                if best_thought.is_final and best_thought.value_estimate >= 0.95:
                    logger.info(f"{LogColors.OKGREEN}[SUCCESS] Found solution with score {best_thought.value_estimate:.2f}{LogColors.ENDC}")
                    return self._create_solution(best_thought), tree, similar_examples
            
            # best attempt if no solution (usually due to errors)
            best_state = max(tree.states.values(), key=lambda x: x.value_estimate)
            logger.info(f"{LogColors.WARNING}[FALLBACK] Returning best partial solution with score {best_state.value_estimate:.2f}{LogColors.ENDC}")
            return self._create_solution(best_state), tree, similar_examples
            
        except Exception as e:
            logger.error(f"{LogColors.FAIL}[ERROR] Problem solving failed: {str(e)}{LogColors.ENDC}")
            raise

    def _create_solution(self, state: ThoughtState) -> Solution:
        """Create a Solution object from a ThoughtState"""
        logger.info(f"{LogColors.OKBLUE}[CREATE] Building final solution from state {state.state_id}{LogColors.ENDC}")
        
        # get numeric
        numeric_values = state.numeric_values or {}
        if state.result:
            try:
                result_value = float(state.result)
                numeric_values['final_result'] = result_value
                logger.info(f"{LogColors.OKGREEN}[FINAL] Numeric result: {result_value}{LogColors.ENDC}")
            except (ValueError, TypeError):
                pass
        
        return Solution(
            explanation=state.reasoning,
            numeric_values=numeric_values,
            symbolic_result=state.result
        )