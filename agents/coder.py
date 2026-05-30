import re
import subprocess
from utils.gemini_client import GeminiClient
from utils.logger import get_logger

logger = get_logger(__name__)

class CoderAgent:
    FORBIDDEN = [
        "os.system", "subprocess.run", "subprocess.call", "subprocess.Popen",
        "eval(", "exec(", "__import__", "socket.", "urllib.request",
        "open(", "os.remove", "os.rmdir", "shutil"
    ]

    def __init__(self):
        self.gemini_client = GeminiClient()

    async def run(self, task: str, context: str) -> str:
        logger.info(f"CoderAgent starting task: '{task}'")
        
        system_prompt = (
            "You are an expert Python engineer. Write clean, well-commented, working Python code.\n"
            f"Task: {task}\n"
            "Return ONLY the code inside a ```python code block. No explanations outside the block."
        )
        
        code = ""
        try:
            raw_response = await self.gemini_client.generate(
                system_prompt=system_prompt,
                user_message=task,
                temperature=0.2
            )
            
            # Extract code block
            code_match = re.search(r"```python\n(.*?)\n```", raw_response, re.DOTALL)
            if not code_match:
                code_match = re.search(r"```\n(.*?)\n```", raw_response, re.DOTALL)
                
            if not code_match:
                code = raw_response.strip()
            else:
                code = code_match.group(1)

            # Safety check
            found_restricted = [pat for pat in self.FORBIDDEN if pat in code]
            if found_restricted:
                logger.warning(f"Safety warning: code contained restricted statements: {found_restricted}")
                return (
                    f"⚠️ Code generated but not executed (contains restricted operations: {', '.join(found_restricted)}).\n"
                    f"Review before running locally.\n\n"
                    f"```python\n{code}\n```"
                )
                
            logger.info("Code safety check passed. Running subprocess...")
            
            # Execute code
            res = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            stdout_str = res.stdout.strip()
            stderr_str = res.stderr.strip()
            
            output_blocks = [
                "✅ CODE GENERATED & EXECUTED\n",
                f"```python\n{code}\n```\n",
                "OUTPUT:",
                stdout_str if stdout_str else "[No output on stdout]"
            ]
            
            if stderr_str:
                output_blocks.append(f"\nERRORS:\n{stderr_str}")
                
            return "\n".join(output_blocks)
            
        except subprocess.TimeoutExpired:
            logger.error("Execution timed out after 30 seconds.")
            return f"❌ Code execution timed out after 30 seconds.\n\n```python\n{code}\n```"
        except Exception as e:
            logger.error(f"Failed inside CoderAgent: {e}")
            return f"Error executing generated code: {e}"
