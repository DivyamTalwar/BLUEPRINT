import docker
from typing import Dict, Any, Optional, Tuple
import tempfile
import shutil
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DockerRunner:
    """Execute code safely in Docker containers"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Docker runner

        Args:
            config: Docker configuration
        """
        self.config = config
        self.timeout = config.get("timeout", 300)
        self.memory_limit = config.get("memory_limit", "1g")
        self.cpu_limit = config.get("cpu_limit", 1.0)
        self.base_image = config.get("base_image", "python:3.11-slim")

        # Try to initialize Docker client (non-blocking)
        self.client = None
        self._docker_available = False
        try:
            self.client = docker.from_env()
            self.client.ping()
            self._docker_available = True
            logger.info("DockerRunner initialized with base image: %s", self.base_image)
        except Exception as e:
            logger.warning("Docker not available: %s (TDD will be limited)", str(e))
            self._docker_available = False

    def execute_tests(
        self,
        code_dir: str,
        test_command: str = "pytest tests/",
        requirements: Optional[list[str]] = None,
    ) -> Tuple[bool, str, str]:
        """
        Execute tests in Docker container

        Args:
            code_dir: Directory containing code and tests
            test_command: Command to run tests
            requirements: Python requirements

        Returns:
            Tuple of (success, stdout, stderr)
        """
        if not self._docker_available:
            logger.warning("Docker not available - cannot execute tests")
            return False, "", "Docker not available"

        temp_dir = None

        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            temp_path = Path(temp_dir)

            # Copy code to temp directory
            code_path = Path(code_dir)
            if code_path.exists():
                shutil.copytree(code_path, temp_path / "app", dirs_exist_ok=True)
            else:
                (temp_path / "app").mkdir(parents=True)

            # Create requirements.txt
            req_file = temp_path / "requirements.txt"
            if requirements:
                req_file.write_text("\n".join(requirements))
            else:
                req_file.write_text("pytest\npytest-cov")

            # Create Dockerfile
            dockerfile = temp_path / "Dockerfile"
            dockerfile_content = f"""
FROM {self.base_image}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/

CMD {test_command}
"""
            dockerfile.write_text(dockerfile_content)

            # Build image
            logger.info("Building Docker image...")
            image, build_logs = self.client.images.build(
                path=str(temp_path), tag="blueprint-test", rm=True
            )

            # Run container
            logger.info("Running tests in container...")
            container = self.client.containers.run(
                image=image.id,
                detach=True,
                mem_limit=self.memory_limit,
                cpu_quota=int(self.cpu_limit * 100000),
                remove=False,
            )

            # Wait for completion
            try:
                result = container.wait(timeout=self.timeout)
                exit_code = result["StatusCode"]

                # Get logs
                stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8")

                success = exit_code == 0

                logger.info(
                    "Test execution completed: success=%s, exit_code=%d",
                    success,
                    exit_code,
                )

                return success, stdout, stderr

            except docker.errors.DockerException as e:
                logger.error("Container timeout or error: %s", str(e))
                container.stop(timeout=1)
                return False, "", f"Container timeout: {str(e)}"

            finally:
                # Cleanup container
                try:
                    container.remove(force=True)
                except:
                    pass

        except Exception as e:
            logger.error("Docker execution error: %s", str(e))
            return False, "", str(e)

        finally:
            # Cleanup temp directory
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def cleanup_images(self, tag_pattern: str = "blueprint-"):
        """
        Cleanup Docker images

        Args:
            tag_pattern: Pattern to match image tags
        """
        try:
            images = self.client.images.list()
            for image in images:
                for tag in image.tags:
                    if tag_pattern in tag:
                        logger.info("Removing image: %s", tag)
                        self.client.images.remove(image.id, force=True)
        except Exception as e:
            logger.error("Error cleaning up images: %s", str(e))

    def is_docker_available(self) -> bool:
        """Check if Docker is available"""
        return self._docker_available

    def run_code(
        self,
        code: str,
        test_mode: bool = False,
        timeout: Optional[int] = None,
        requirements: Optional[list[str]] = None
    ) -> Dict[str, Any]:
        """
        Run code (with optional tests) in Docker container.

        Args:
            code: Python code to execute
            test_mode: If True, runs with pytest
            timeout: Timeout in seconds (default: self.timeout)
            requirements: Additional Python packages

        Returns:
            Dict with success, output, error, exit_code
        """
        # Check Docker availability first
        if not self._docker_available or self.client is None:
            logger.warning("Docker not available - cannot run code")
            return {
                "success": False,
                "output": "",
                "error": "Docker not available",
                "exit_code": -1,
                "error_type": "docker_unavailable"
            }

        temp_dir = None
        timeout = timeout or self.timeout

        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            temp_path = Path(temp_dir)

            # Write code to file
            code_file = temp_path / "code.py"
            code_file.write_text(code)

            # Create requirements.txt
            req_file = temp_path / "requirements.txt"
            base_reqs = ["pytest", "pytest-timeout"] if test_mode else []
            if requirements:
                base_reqs.extend(requirements)
            req_file.write_text("\n".join(base_reqs))

            # Create Dockerfile
            dockerfile = temp_path / "Dockerfile"
            if test_mode:
                cmd = '["pytest", "code.py", "-v", "--tb=short"]'
            else:
                cmd = '["python", "code.py"]'

            dockerfile_content = f"""
FROM {self.base_image}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code.py .

CMD {cmd}
"""
            dockerfile.write_text(dockerfile_content)

            # Build image
            logger.info("Building Docker image for code execution...")
            image, build_logs = self.client.images.build(
                path=str(temp_dir), tag="blueprint-code-exec", rm=True, quiet=True
            )

            # Run container
            logger.info("Running code in container...")
            container = self.client.containers.run(
                image=image.id,
                detach=True,
                mem_limit=self.memory_limit,
                cpu_quota=int(self.cpu_limit * 100000),
                remove=False,
            )

            # Wait for completion
            try:
                result = container.wait(timeout=timeout)
                exit_code = result["StatusCode"]

                # Get logs
                stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8")

                success = exit_code == 0

                logger.info("Code execution completed: success=%s", success)

                return {
                    "success": success,
                    "output": stdout,
                    "error": stderr,
                    "exit_code": exit_code,
                    "error_type": "none" if success else "test_failure"
                }

            except docker.errors.DockerException as e:
                logger.error("Container timeout or error: %s", str(e))
                container.stop(timeout=1)
                return {
                    "success": False,
                    "output": "",
                    "error": f"Container timeout: {str(e)}",
                    "exit_code": -1,
                    "error_type": "timeout"
                }

            finally:
                # Cleanup container
                try:
                    container.remove(force=True)
                except:
                    pass

                # Cleanup image
                try:
                    self.client.images.remove(image.id, force=True)
                except:
                    pass

        except Exception as e:
            logger.error("Docker execution error: %s", str(e))
            error_msg = str(e)

            # Classify error type
            if "No such image" in error_msg or "image not found" in error_msg:
                error_type = "image_error"
            elif "Cannot connect" in error_msg or "connection refused" in error_msg:
                error_type = "docker_unavailable"
            elif "permission denied" in error_msg.lower():
                error_type = "permission_error"
            else:
                error_type = "docker_error"

            return {
                "success": False,
                "output": "",
                "error": error_msg,
                "exit_code": -1,
                "error_type": error_type
            }

        finally:
            # Cleanup temp directory
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
