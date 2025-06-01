# AI
from openai import AsyncOpenAI

# API
from router_api.router_api import RouteLLM, get_api_key
from fastapi.testclient import TestClient
from router_api.router_api import app

# Other
import unittest
import os

class TestRouteLLM(unittest.TestCase):
    
    def setUp(self) -> None:
        """
        Set up the test environment
        """

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self.skipTest("OPENAI_API_KEY environment variable not set")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.query = "What is Python?"
        self.temperature = 0.7
        
    def test_init_without_file(self) -> None:
        """
        Test the initialization of the RouteLLM class without a file
        """

        router = RouteLLM(self.query, self.temperature, self.client)
        self.assertEqual(router.query, self.query)
        self.assertEqual(router.temperature, self.temperature)
        self.assertFalse(router.file_added)
        self.assertIsNone(router.file_type)
        
    def test_init_with_file(self) -> None:
        """
        Test the initialization of the RouteLLM class with a file
        """

        file_path = "test.jpg"
        router = RouteLLM(self.query, self.temperature, self.client, file_path)
        self.assertTrue(router.file_added)
        self.assertEqual(router.file_type, "image")
        
    def test_get_file_type_image(self) -> None:
        router = RouteLLM(self.query, self.temperature, self.client, "test.png")
        self.assertEqual(router.file_type, "image")
        
    def test_get_file_type_pdf(self) -> None:
        """
        Test the get_file_type method for a PDF file
        """

        router = RouteLLM(self.query, self.temperature, self.client, "test.pdf")
        self.assertEqual(router.file_type, "pdf")
        
    def test_get_file_type_unsupported(self) -> None:
        """
        Test the get_file_type method for an unsupported file type
        """

        router = RouteLLM(self.query, self.temperature, self.client, "test.txt")
        self.assertEqual(router.file_type, "unsupported")
        
    def test_xml_detection(self) -> None:
        """
        Test the XML detection functionality when XML is present
        """

        xml_query = "Parse this <xml>content</xml>"
        router = RouteLLM(xml_query, self.temperature, self.client)
        self.assertTrue(router.xml_true)
        
    def test_no_xml_detection(self) -> None:
        """
        Test the XML detection functionality when no XML is present
        """

        router = RouteLLM(self.query, self.temperature, self.client)
        self.assertFalse(router.xml_true)

class TestAsyncRouteLLM(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self) -> None:
        """
        Set up the test environment
        """

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self.skipTest("OPENAI_API_KEY environment variable not set")
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.query = "What is Python in one sentence?"
        self.temperature = 0.7
        
    async def test_get_evaluation_response(self) -> None:
        """
        Test the get_evaluation_response method
        """

        router = RouteLLM(self.query, self.temperature, self.client)
        result = await router.get_evaluation_response()
        
        self.assertIn(result, ["gpt-4.1-nano", "gpt-4.1", "o4-mini"])
        
    async def test_get_gpt_response(self) -> None:
        """
        Test the get_gpt_response method
        """

        router = RouteLLM(self.query, self.temperature, self.client)
        result = await router.get_gpt_response("gpt-4o-mini")
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        
    async def test_orchestrate_response(self) -> None:
        """
        Test the orchestrate_response method
        """

        router = RouteLLM(self.query, self.temperature, self.client)
        result = await router.orchestrate_response()
        
        self.assertIn("response", result)
        self.assertIn("selected_model", result)
        self.assertIsInstance(result["response"], str)
        self.assertGreater(len(result["response"]), 0)
        
    async def test_prepare_messages_no_file(self) -> None:
        """
        Test the prepare_messages method when no file is present
        """

        router = RouteLLM(self.query, self.temperature, self.client)
        messages = await router._prepare_messages()
        
        expected = [{"role": "user", "content": self.query}]
        self.assertEqual(messages, expected)


class TestGetApiKey(unittest.TestCase):
    
    def test_get_api_key_from_request(self) -> None:
        """
        Test the get_api_key_from_request method
        """

        result = get_api_key("test_key", None)
        self.assertEqual(result, "test_key")
        
    def test_get_api_key_from_header(self) -> None:
        """
        Test the get_api_key_from_header method
        """

        result = get_api_key(None, "header_key")
        self.assertEqual(result, "header_key")
        
    def test_get_api_key_request_priority(self) -> None:
        """
        Test the get_api_key_request_priority method
        """

        result = get_api_key("request_key", "header_key")
        self.assertEqual(result, "request_key")

class TestAPIEndpoints(unittest.TestCase):
    
    def setUp(self) -> None:
        """
        Set up the test environment
        """

        self.client = TestClient(app)
        
    def test_root_endpoint(self) -> None:
        """
        Test the root endpoint
        """

        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "LLM Router API")
        
    def test_health_endpoint(self) -> None:
        """
        Test the health endpoint
        """

        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        
    def test_models_endpoint(self) -> None:
        """
        Test the models endpoint
        """

        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        self.assertIn("gpt-4.1-nano", data["models"])
        self.assertIn("gpt-4.1", data["models"])
        self.assertIn("o4-mini", data["models"])

    def test_route_endpoint_with_api_key(self) -> None:
        """
        Test the route endpoint with an API key
        """

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.skipTest("OPENAI_API_KEY environment variable not set")
            
        response = self.client.post(
            "/route",
            json={
                "prompt": "What is 2+2?",
                "temperature": 0.1,
                "openai_api_key": api_key
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        self.assertIn("selected_model", data)


if __name__ == "__main__":
    unittest.main()