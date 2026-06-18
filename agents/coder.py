import ast
import re
import subprocess
from utils.gemini_client import GeminiClient
from utils.logger import get_logger

logger = get_logger(__name__)


class CoderAgent:
    def __init__(self):
        self.gemini_client = GeminiClient()

    def _is_code_safe(self, code: str) -> tuple[bool, list[str]]:
        """Use AST parsing for real safety analysis instead of string matching."""
        violations = []
        
        # 1. AST-level analysis
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # Block subprocess, os.system, eval, exec
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Attribute):
                        # Build dotted name like "os.system" or "subprocess.run"
                        value_id = getattr(func.value, "id", "")
                        full = f"{value_id}.{func.attr}"
                        dangerous = [
                            "subprocess.run", "subprocess.call", "subprocess.Popen",
                            "subprocess.check_output", "subprocess.check_call",
                            "os.system", "os.popen", "os.remove", "os.rmdir",
                            "os.unlink", "os.rename", "shutil.rmtree", "shutil.move"
                        ]
                        for d in dangerous:
                            if d in full:
                                violations.append(f"Forbidden call: {full}")
                    elif isinstance(func, ast.Name):
                        if func.id in ["eval", "exec", "__import__", "compile"]:
                            violations.append(f"Forbidden builtin: {func.id}")
                        if func.id == "open":
                            violations.append("Forbidden: open() file access")
        except SyntaxError as e:
            violations.append(f"Code has syntax error: {e}")
        
        # 2. Regex for obfuscation patterns
        obfuscation_patterns = [
            (r"__builtins__", "Accessing __builtins__"),
            (r"getattr\s*\(", "Dynamic attribute access via getattr"),
            (r"importlib", "Dynamic import via importlib"),
            (r"ctypes", "Low-level ctypes access"),
            (r"sys\.modules", "Module table manipulation"),
        ]
        for pattern, desc in obfuscation_patterns:
            if re.search(pattern, code):
                violations.append(f"Suspicious pattern: {desc}")
        
        return len(violations) == 0, violations

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

            # Safety check via AST analysis
            is_safe, violations = self._is_code_safe(code)
            if not is_safe:
                logger.warning(f"Safety violations detected: {violations}")
                return (
                    f"⚠️ Code generated but not executed (safety violations detected).\n"
                    f"Violations: {', '.join(violations)}\n"
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
