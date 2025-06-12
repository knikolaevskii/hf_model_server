#!/usr/bin/env python3
"""
Comprehensive Model Tester for Hugging Face Server
Tests different models with various prompts and parameters
"""

import requests
import json
import time
import argparse
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import statistics

@dataclass
class TestCase:
    name: str
    prompt: str
    expected_keywords: List[str] = None
    max_tokens: int = 100
    temperature: float = None
    top_p: float = None
    description: str = ""

@dataclass
class TestResult:
    test_name: str
    model_name: str
    prompt: str
    response: str
    response_time: float
    success: bool
    error: str = None
    tokens_used: int = 0
    keywords_found: List[str] = None

class ModelTester:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip('/')
        self.results: List[TestResult] = []
        
        # Define comprehensive test cases
        self.test_cases = [
            TestCase(
                name="basic_greeting",
                prompt="Hello! How are you today?",
                expected_keywords=["hello", "hi", "good", "fine", "well"],
                description="Basic greeting and conversation"
            ),
            TestCase(
                name="python_question",
                prompt="What is Python programming language?",
                expected_keywords=["python", "programming", "language", "code"],
                description="Technical question about Python"
            ),
            TestCase(
                name="sql_generation",
                prompt="Generate SQL to select all users from a table called 'users' where age > 18:",
                expected_keywords=["SELECT", "FROM", "users", "WHERE", "age", ">", "18"],
                description="SQL code generation"
            ),
            TestCase(
                name="story_creative",
                prompt="Write a short story about a robot learning to paint:",
                expected_keywords=["robot", "paint", "learn", "art", "story"],
                max_tokens=200,
                temperature=0.8,
                description="Creative writing task"
            ),
            TestCase(
                name="math_problem",
                prompt="Solve this math problem step by step: What is 15% of 240?",
                expected_keywords=["15", "240", "36", "percent", "%"],
                description="Mathematical problem solving"
            ),
            TestCase(
                name="explanation_science",
                prompt="Explain photosynthesis in simple terms:",
                expected_keywords=["photosynthesis", "plant", "sunlight", "oxygen", "carbon"],
                description="Scientific explanation"
            ),
            TestCase(
                name="code_debug",
                prompt="Debug this Python code: for i in range(10) print(i)",
                expected_keywords=["colon", ":", "syntax", "error", "for", "print"],
                description="Code debugging task"
            ),
            TestCase(
                name="creative_conservative",
                prompt="List 5 benefits of reading books:",
                expected_keywords=["reading", "books", "benefit", "knowledge", "improve"],
                temperature=0.3,
                description="Conservative generation for factual content"
            ),
            TestCase(
                name="conversation_context",
                prompt="I'm feeling stressed about work. What advice do you have?",
                expected_keywords=["stress", "work", "advice", "relax", "help"],
                description="Conversational support"
            ),
            TestCase(
                name="technical_explanation",
                prompt="Explain the difference between machine learning and deep learning:",
                expected_keywords=["machine learning", "deep learning", "neural", "algorithm", "data"],
                max_tokens=150,
                description="Technical concept explanation"
            )
        ]
    
    def check_server_health(self) -> bool:
        """Check if the server is running and healthy"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Server is healthy")
                print(f"   Model loaded: {health_data.get('model_loaded', 'Unknown')}")
                if health_data.get('model_info'):
                    model_info = health_data['model_info']
                    print(f"   Current model: {model_info.get('name', 'Unknown')}")
                    print(f"   Parameters: {model_info.get('total_params', 0)/1e6:.1f}M")
                return True
            else:
                print(f"❌ Server unhealthy: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
    
    def load_model(self, model_name: str, force_reload: bool = False) -> bool:
        """Load a model on the server"""
        try:
            print(f"🔄 Loading model: {model_name}")
            response = requests.post(
                f"{self.server_url}/load_model",
                params={"model_name": model_name, "force_reload": force_reload},
                timeout=300  # 5 minutes timeout for model loading
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Model loaded successfully")
                if result.get('model_info'):
                    info = result['model_info']
                    print(f"   Load time: {info.get('load_time', 0):.2f}s")
                    print(f"   Parameters: {info.get('total_params', 0)/1e6:.1f}M")
                return True
            else:
                print(f"❌ Failed to load model: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def run_single_test(self, test_case: TestCase, model_name: str) -> TestResult:
        """Run a single test case"""
        print(f"  🧪 Running: {test_case.name}")
        
        # Prepare the request
        messages = [{"role": "user", "content": test_case.prompt}]
        
        request_data = {
            "model": model_name,
            "messages": messages,
            "max_tokens": test_case.max_tokens
        }
        
        # Add optional parameters only if specified
        if test_case.temperature is not None:
            request_data["temperature"] = test_case.temperature
        if test_case.top_p is not None:
            request_data["top_p"] = test_case.top_p
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=request_data,
                timeout=60,
                headers={"Content-Type": "application/json"}
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result_data = response.json()
                generated_text = result_data["choices"][0]["message"]["content"]
                usage = result_data.get("usage", {})
                
                # Check for expected keywords
                keywords_found = []
                if test_case.expected_keywords:
                    generated_lower = generated_text.lower()
                    for keyword in test_case.expected_keywords:
                        if keyword.lower() in generated_lower:
                            keywords_found.append(keyword)
                
                return TestResult(
                    test_name=test_case.name,
                    model_name=model_name,
                    prompt=test_case.prompt,
                    response=generated_text,
                    response_time=response_time,
                    success=True,
                    tokens_used=usage.get("total_tokens", 0),
                    keywords_found=keywords_found
                )
            else:
                return TestResult(
                    test_name=test_case.name,
                    model_name=model_name,
                    prompt=test_case.prompt,
                    response="",
                    response_time=response_time,
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}"
                )
                
        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            return TestResult(
                test_name=test_case.name,
                model_name=model_name,
                prompt=test_case.prompt,
                response="",
                response_time=response_time,
                success=False,
                error=str(e)
            )
    
    def test_model(self, model_name: str, test_cases: List[str] = None) -> List[TestResult]:
        """Test a specific model with selected or all test cases"""
        print(f"\n🤖 Testing model: {model_name}")
        print("=" * 60)
        
        # Filter test cases if specific ones are requested
        if test_cases:
            filtered_tests = [tc for tc in self.test_cases if tc.name in test_cases]
            if not filtered_tests:
                print(f"❌ No matching test cases found: {test_cases}")
                return []
        else:
            filtered_tests = self.test_cases
        
        model_results = []
        
        for i, test_case in enumerate(filtered_tests, 1):
            print(f"\n[{i}/{len(filtered_tests)}] {test_case.description}")
            result = self.run_single_test(test_case, model_name)
            model_results.append(result)
            
            if result.success:
                print(f"     ✅ Success ({result.response_time:.2f}s)")
                print(f"     📝 Response: {result.response[:100]}{'...' if len(result.response) > 100 else ''}")
                
                if result.keywords_found and test_case.expected_keywords:
                    keyword_score = len(result.keywords_found) / len(test_case.expected_keywords)
                    print(f"     🎯 Keywords found: {len(result.keywords_found)}/{len(test_case.expected_keywords)} ({keyword_score:.1%})")
                
            else:
                print(f"     ❌ Failed: {result.error}")
            
            # Small delay between tests
            time.sleep(0.5)
        
        return model_results
    
    def compare_models(self, models: List[str], test_cases: List[str] = None):
        """Compare multiple models on the same test cases"""
        print(f"\n🏆 Comparing Models: {', '.join(models)}")
        print("=" * 80)
        
        all_results = {}
        
        for model in models:
            # Load the model
            if not self.load_model(model):
                print(f"❌ Skipping {model} - failed to load")
                continue
            
            # Wait a bit for model to be ready
            time.sleep(2)
            
            # Test the model
            results = self.test_model(model, test_cases)
            all_results[model] = results
            self.results.extend(results)
        
        # Generate comparison report
        self.generate_comparison_report(all_results)
    
    def generate_comparison_report(self, all_results: Dict[str, List[TestResult]]):
        """Generate a detailed comparison report"""
        print(f"\n📊 COMPARISON REPORT")
        print("=" * 80)
        
        if not all_results:
            print("No results to compare")
            return
        
        # Overall statistics
        print("\n📈 OVERALL STATISTICS")
        print("-" * 40)
        
        for model, results in all_results.items():
            successful_tests = [r for r in results if r.success]
            failed_tests = [r for r in results if not r.success]
            
            if successful_tests:
                avg_response_time = statistics.mean([r.response_time for r in successful_tests])
                total_tokens = sum([r.tokens_used for r in successful_tests])
            else:
                avg_response_time = 0
                total_tokens = 0
            
            print(f"\n🤖 {model}:")
            print(f"   ✅ Successful: {len(successful_tests)}/{len(results)} ({len(successful_tests)/len(results):.1%})")
            print(f"   ⏱️  Avg Response Time: {avg_response_time:.2f}s")
            print(f"   🎯 Total Tokens Used: {total_tokens}")
            
            if failed_tests:
                print(f"   ❌ Failed Tests: {[r.test_name for r in failed_tests]}")
        
        # Test-by-test comparison
        print(f"\n🔍 TEST-BY-TEST COMPARISON")
        print("-" * 40)
        
        # Get all test names
        all_test_names = set()
        for results in all_results.values():
            all_test_names.update([r.test_name for r in results])
        
        for test_name in sorted(all_test_names):
            print(f"\n📝 {test_name}:")
            
            for model, results in all_results.items():
                test_result = next((r for r in results if r.test_name == test_name), None)
                
                if test_result:
                    if test_result.success:
                        status = f"✅ {test_result.response_time:.2f}s"
                        if test_result.keywords_found:
                            # Get the original test case to calculate keyword score
                            original_test = next((tc for tc in self.test_cases if tc.name == test_name), None)
                            if original_test and original_test.expected_keywords:
                                keyword_score = len(test_result.keywords_found) / len(original_test.expected_keywords)
                                status += f" (🎯{keyword_score:.1%})"
                    else:
                        status = f"❌ {test_result.error[:50]}..."
                else:
                    status = "⚪ Not tested"
                
                print(f"   {model}: {status}")
    
    def save_results(self, filename: str = None):
        """Save test results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_test_results_{timestamp}.json"
        
        # Convert results to dict for JSON serialization
        results_dict = []
        for result in self.results:
            results_dict.append({
                "test_name": result.test_name,
                "model_name": result.model_name,
                "prompt": result.prompt,
                "response": result.response,
                "response_time": result.response_time,
                "success": result.success,
                "error": result.error,
                "tokens_used": result.tokens_used,
                "keywords_found": result.keywords_found,
                "timestamp": datetime.now().isoformat()
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filename}")
    
    def list_available_tests(self):
        """List all available test cases"""
        print("\n📋 AVAILABLE TEST CASES")
        print("=" * 50)
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"{i:2d}. {test_case.name}")
            print(f"    📝 {test_case.description}")
            print(f"    💬 \"{test_case.prompt[:60]}{'...' if len(test_case.prompt) > 60 else ''}\"")
            if test_case.expected_keywords:
                print(f"    🎯 Keywords: {', '.join(test_case.expected_keywords[:5])}")
            print()

def main():
    parser = argparse.ArgumentParser(description="Test Hugging Face Model Server")
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                       help="Server URL (default: http://localhost:8000)")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                       help="Models to test (space-separated)")
    parser.add_argument("--tests", type=str, nargs="*",
                       help="Specific test cases to run (default: all)")
    parser.add_argument("--list-tests", action="store_true",
                       help="List available test cases and exit")
    parser.add_argument("--save", type=str,
                       help="Save results to file")
    parser.add_argument("--no-load", action="store_true",
                       help="Skip model loading (assume models are already loaded)")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ModelTester(args.server)
    
    # List tests if requested
    if args.list_tests:
        tester.list_available_tests()
        return
    
    # Check server health
    if not tester.check_server_health():
        print("❌ Server is not healthy. Please check your server.")
        sys.exit(1)
    
    print(f"\n🚀 Starting model testing...")
    print(f"Server: {args.server}")
    print(f"Models: {', '.join(args.models)}")
    if args.tests:
        print(f"Tests: {', '.join(args.tests)}")
    else:
        print(f"Tests: All ({len(tester.test_cases)} tests)")
    
    # Run tests
    if len(args.models) == 1:
        # Single model testing
        model = args.models[0]
        if not args.no_load:
            if not tester.load_model(model):
                print(f"❌ Failed to load model: {model}")
                sys.exit(1)
        
        results = tester.test_model(model, args.tests)
        tester.results.extend(results)
    else:
        # Multi-model comparison
        tester.compare_models(args.models, args.tests)
    
    # Save results if requested
    if args.save:
        tester.save_results(args.save)
    
    print(f"\n🎉 Testing completed! Total tests: {len(tester.results)}")

if __name__ == "__main__":
    main()