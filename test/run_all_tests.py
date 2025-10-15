#!/usr/bin/env python3
"""
Comprehensive Test Runner

Executes all test suites for the adaptive signal system and provides unified reporting:
- Adaptive Signal System Tests
- Market Regime Detection Tests  
- Edge Cases and Error Handling Tests
- Enhanced Signal System Tests (if available)

Provides detailed reporting, performance metrics, and overall system health assessment.
"""

import sys
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))


class TestRunner:
    """Comprehensive test runner for all signal system tests"""
    
    def __init__(self):
        """Initialize test runner"""
        self.test_dir = os.path.dirname(__file__)
        self.results = {}
        self.start_time = None
        self.total_tests = 0
        self.passed_tests = 0
        
    def run_test_file(self, test_file: str, description: str) -> Tuple[bool, Dict[str, Any]]:
        """Run a single test file and capture results"""
        print(f"\\n{'='*70}")
        print(f"🧪 {description}")
        print(f"{'='*70}")
        
        test_path = os.path.join(self.test_dir, test_file)
        
        if not os.path.exists(test_path):
            print(f"❌ Test file not found: {test_file}")
            return False, {
                'status': 'not_found',
                'error': f"Test file {test_file} not found",
                'duration': 0,
                'tests_run': 0,
                'tests_passed': 0
            }
        
        start_time = time.time()
        
        try:
            # Run the test file with proper encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run(
                [sys.executable, test_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=os.path.dirname(test_path),
                env=env
            )
            
            duration = time.time() - start_time
            
            # Parse output for test results
            output = result.stdout
            error_output = result.stderr
            
            # Extract test counts from output
            tests_run = 0
            tests_passed = 0
            
            # Look for test result patterns
            if "TEST RESULTS:" in output:
                for line in output.split('\\n'):
                    if "TEST RESULTS:" in line:
                        # Extract pattern like "TEST RESULTS: 4/4 tests passed"
                        parts = line.split("TEST RESULTS:")[-1].strip()
                        if "/" in parts and "tests passed" in parts:
                            try:
                                passed_total = parts.split()[0]
                                tests_passed = int(passed_total.split('/')[0])
                                tests_run = int(passed_total.split('/')[1])
                            except (ValueError, IndexError):
                                pass
            
            # Determine success
            success = result.returncode == 0 and tests_passed == tests_run and tests_run > 0
            
            test_result = {
                'status': 'success' if success else 'failed',
                'return_code': result.returncode,
                'duration': duration,
                'tests_run': tests_run,
                'tests_passed': tests_passed,
                'output': output,
                'error': error_output if error_output else None
            }
            
            # Print summary
            if success:
                print(f"✅ {description}: {tests_passed}/{tests_run} tests passed ({duration:.2f}s)")
            else:
                print(f"❌ {description}: {tests_passed}/{tests_run} tests passed ({duration:.2f}s)")
                if error_output:
                    print(f"   Error: {error_output[:200]}...")
            
            return success, test_result
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ {description}: Exception occurred - {e}")
            
            return False, {
                'status': 'exception',
                'error': str(e),
                'duration': duration,
                'tests_run': 0,
                'tests_passed': 0
            }
    
    def run_all_tests(self) -> bool:
        """Run all available test suites"""
        print("🚀 COMPREHENSIVE SIGNAL SYSTEM TEST SUITE")
        print("=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.start_time = time.time()
        
        # Define test suites
        test_suites = [
            ("test_adaptive_signal_system.py", "Adaptive Signal System Core Tests"),
            ("test_market_regime_detection.py", "Market Regime Detection Tests"),
            ("test_edge_cases.py", "Edge Cases and Error Handling Tests"),
            ("test_enhanced_signal_system.py", "Enhanced Signal System Tests"),
        ]
        
        all_passed = True
        
        # Run each test suite
        for test_file, description in test_suites:
            success, result = self.run_test_file(test_file, description)
            self.results[test_file] = result
            
            if success:
                self.passed_tests += result['tests_passed']
            self.total_tests += result['tests_run']
            
            if not success:
                all_passed = False
        
        # Generate final report
        self.generate_final_report()
        
        return all_passed
    
    def generate_final_report(self) -> None:
        """Generate comprehensive final report"""
        total_duration = time.time() - self.start_time
        
        print("\\n" + "=" * 70)
        print("📊 COMPREHENSIVE TEST RESULTS")
        print("=" * 70)
        
        # Overall summary
        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Overall Results: {self.passed_tests}/{self.total_tests} tests passed")
        
        if self.passed_tests == self.total_tests and self.total_tests > 0:
            print("🎉 ALL TESTS PASSED!")
            overall_status = "SUCCESS"
        else:
            print("🔧 SOME TESTS FAILED")
            overall_status = "FAILED"
        
        print("\\n📋 Test Suite Breakdown:")
        print("-" * 50)
        
        # Detailed breakdown
        for test_file, result in self.results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            suite_name = test_file.replace('test_', '').replace('.py', '').replace('_', ' ').title()
            
            print(f"{status_icon} {suite_name}")
            print(f"   Tests: {result['tests_passed']}/{result['tests_run']} passed")
            print(f"   Duration: {result['duration']:.2f}s")
            
            if result['status'] != 'success':
                if result.get('error'):
                    error_preview = result['error'][:100] + "..." if len(result['error']) > 100 else result['error']
                    print(f"   Error: {error_preview}")
        
        # Performance metrics
        print("\\n⚡ Performance Metrics:")
        print("-" * 30)
        
        fastest_suite = min(self.results.items(), key=lambda x: x[1]['duration'])
        slowest_suite = max(self.results.items(), key=lambda x: x[1]['duration'])
        
        print(f"Fastest Suite: {fastest_suite[0]} ({fastest_suite[1]['duration']:.2f}s)")
        print(f"Slowest Suite: {slowest_suite[0]} ({slowest_suite[1]['duration']:.2f}s)")
        print(f"Average Suite Duration: {total_duration / len(self.results):.2f}s")
        
        if self.total_tests > 0:
            print(f"Tests per Second: {self.total_tests / total_duration:.1f}")
        
        # System health assessment
        print("\\n🏥 System Health Assessment:")
        print("-" * 35)
        
        health_score = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        if health_score >= 95:
            health_status = "EXCELLENT"
            health_icon = "🟢"
        elif health_score >= 80:
            health_status = "GOOD"
            health_icon = "🟡"
        elif health_score >= 60:
            health_status = "FAIR"
            health_icon = "🟠"
        else:
            health_status = "POOR"
            health_icon = "🔴"
        
        print(f"{health_icon} Overall Health: {health_status} ({health_score:.1f}%)")
        
        # Recommendations
        print("\\n💡 Recommendations:")
        print("-" * 20)
        
        if health_score == 100:
            print("✨ System is in excellent condition!")
            print("✨ All adaptive signal components are working correctly")
            print("✨ Ready for production deployment")
        elif health_score >= 80:
            print("👍 System is in good condition")
            print("🔧 Address failing tests before production deployment")
            print("📈 Consider performance optimizations for slower suites")
        else:
            print("⚠️  System needs attention")
            print("🚨 Critical issues detected - do not deploy to production")
            print("🔧 Review and fix failing tests immediately")
            print("📊 Consider system architecture review")
        
        # Save detailed results to file
        self.save_results_to_file(overall_status, health_score)
        
        print("\\n" + "=" * 70)
        print(f"📁 Detailed results saved to: test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        print("=" * 70)
    
    def save_results_to_file(self, overall_status: str, health_score: float) -> None:
        """Save detailed test results to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"test_results_{timestamp}.json"
        filepath = os.path.join(self.test_dir, filename)
        
        detailed_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'health_score': health_score,
            'total_duration': time.time() - self.start_time,
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'test_suites': self.results,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'test_directory': self.test_dir
            }
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️  Could not save results to file: {e}")


def main():
    """Main entry point"""
    runner = TestRunner()
    success = runner.run_all_tests()
    
    if success:
        print("\\n🎯 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\\n⚠️  Some tests failed. Please review and address issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()