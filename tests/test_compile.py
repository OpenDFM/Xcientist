import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Mock 'agents' module before importing anything that uses it
mock_agents = MagicMock()
# Define function_tool as a pass-through decorator
def function_tool_decorator(func):
    return func
mock_agents.function_tool = function_tool_decorator
sys.modules["agents"] = mock_agents

# Add project root and src to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

# Import the module to test
from src.agents.paper_agent.tools.compile import compile_and_vlm_review

class TestCompileAndVlmReview(unittest.TestCase):

    @patch("src.agents.paper_agent.tools.compile.compile_and_vlm_review_impl")
    def test_compile_success(self, mock_impl):
        # Setup mock return value
        mock_impl.return_value = {
            "compile_success": True,
            "pdf_path": "/path/to/paper.pdf",
            "issues": {"errors": [], "warnings": []},
            "vlm_layout_review": {"success": True, "findings": []}
        }

        # Call the function (new signature)
        result = compile_and_vlm_review(
            paper_dir="/tmp/paper",
            artifact_dir="/tmp/artifact",
            main_tex="main.tex"
        )

        # Verify result
        self.assertTrue(result["success"])
        self.assertIn("compile_success=True", result["message"])
        self.assertIn("pdf=/path/to/paper.pdf", result["message"])
        
        # Verify impl called correctly (without extra args)
        mock_impl.assert_called_once()
        args, kwargs = mock_impl.call_args
        self.assertEqual(kwargs["paper_dir"], "/tmp/paper")
        self.assertEqual(kwargs["artifact_dir"], "/tmp/artifact")
        self.assertEqual(kwargs["main_tex"], "main.tex")
        # Ensure removed args are NOT passed
        self.assertNotIn("vlm_mode", kwargs)
        self.assertNotIn("docker_image", kwargs)

    @patch("src.agents.paper_agent.tools.compile.compile_and_vlm_review_impl")
    def test_compile_failure_with_details(self, mock_impl):
        # Setup mock return value for failure with specific errors
        mock_impl.return_value = {
            "compile_success": False,
            "pdf_path": "",
            "issues": {
                "errors": ["! LaTeX Error: File `article.cls' not found.", "! Emergency stop."], 
                "warnings": [],
                "missing_files": ["article.cls"]
            },
            "vlm_layout_review": {}
        }

        # Call the function
        result = compile_and_vlm_review(
            paper_dir="/tmp/paper",
            artifact_dir="/tmp/artifact"
        )

        # Verify result
        self.assertFalse(result["success"])
        self.assertIn("compile_success=False", result["message"])
        # Check if detailed errors are in message and error field
        self.assertIn("File `article.cls' not found", result["error"])
        self.assertIn("Missing files", result["message"])

if __name__ == "__main__":
    unittest.main()
