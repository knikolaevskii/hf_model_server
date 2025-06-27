#!/usr/bin/env python3
"""
Model Memory Monitor Script with Peak Usage Tracking
Loads models, tracks peak RAM and VRAM usage during loading, then unloads them
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil
import gc
import subprocess
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import threading
from collections import defaultdict

class MemoryTracker:
    def __init__(self):
        self.tracking = False
        self.peak_ram = 0
        self.peak_vram = 0
        self.current_ram = 0
        self.current_vram = 0
        self.memory_history = []
        self.start_time = None
        
    def get_current_memory(self):
        """Get current memory usage"""
        # RAM usage
        process = psutil.Process(os.getpid())
        ram_gb = process.memory_info().rss / (1024 ** 3)
        
        # VRAM usage
        vram_gb = 0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1e9
            
        return ram_gb, vram_gb
    
    def start_tracking(self):
        """Start continuous memory tracking"""
        self.tracking = True
        self.start_time = time.time()
        self.peak_ram = 0
        self.peak_vram = 0
        self.memory_history = []
        
        def track_memory():
            while self.tracking:
                ram, vram = self.get_current_memory()
                self.current_ram = ram
                self.current_vram = vram
                
                # Update peaks
                self.peak_ram = max(self.peak_ram, ram)
                self.peak_vram = max(self.peak_vram, vram)
                
                # Store history with timestamp
                elapsed = time.time() - self.start_time
                self.memory_history.append({
                    'timestamp': elapsed,
                    'ram_gb': ram,
                    'vram_gb': vram
                })
                
                time.sleep(0.1)  # Check every 100ms
        
        self.tracking_thread = threading.Thread(target=track_memory, daemon=True)
        self.tracking_thread.start()
    
    def stop_tracking(self):
        """Stop memory tracking"""
        self.tracking = False
        if hasattr(self, 'tracking_thread'):
            self.tracking_thread.join(timeout=1.0)
    
    def get_peak_usage(self):
        """Get peak memory usage during tracking"""
        return self.peak_ram, self.peak_vram
    
    def get_memory_stats(self):
        """Get detailed memory statistics"""
        if not self.memory_history:
            return {}
            
        ram_values = [entry['ram_gb'] for entry in self.memory_history]
        vram_values = [entry['vram_gb'] for entry in self.memory_history]
        
        return {
            'peak_ram_gb': max(ram_values),
            'peak_vram_gb': max(vram_values),
            'min_ram_gb': min(ram_values),
            'min_vram_gb': min(vram_values),
            'avg_ram_gb': sum(ram_values) / len(ram_values),
            'avg_vram_gb': sum(vram_values) / len(vram_values),
            'samples': len(ram_values),
            'duration': self.memory_history[-1]['timestamp'] if self.memory_history else 0
        }

class ModelMemoryMonitor:
    def __init__(self):
        self.device = None
        self.baseline_ram = 0
        self.baseline_vram = 0
        self.results = []
        self.memory_tracker = MemoryTracker()
        
    def get_gpu_memory(self) -> List[tuple]:
        """Get GPU memory usage for all GPUs"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                capture_output=True, text=True, check=True
            )
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                used, total = map(int, line.split(', '))
                gpu_info.append((used, total))
            return gpu_info
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

    def get_ram_usage(self) -> float:
        """Get current RAM usage in GB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)

    def get_vram_usage(self) -> Dict[str, float]:
        """Get VRAM usage in GB"""
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0}
        
        return {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "reserved": torch.cuda.memory_reserved() / 1e9
        }

    def setup_device(self):
        """Setup the device (CUDA or CPU)"""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"🔥 Using CUDA GPU: {torch.cuda.get_device_name()}")
            
            # Show initial GPU memory
            gpu_memory = self.get_gpu_memory()
            if gpu_memory:
                for i, (used, total) in enumerate(gpu_memory):
                    print(f"📊 GPU {i} Initial Memory: {used}MB / {total}MB ({used/total*100:.1f}% used)")
        else:
            self.device = torch.device("cpu")
            print("⚠️  CUDA not available, using CPU")

    def get_baseline_memory(self):
        """Record baseline memory usage"""
        self.baseline_ram = self.get_ram_usage()
        vram_info = self.get_vram_usage()
        self.baseline_vram = vram_info["allocated"]
        
        print(f"📊 Baseline RAM: {self.baseline_ram:.2f} GB")
        print(f"📊 Baseline VRAM: {self.baseline_vram:.2f} GB")

    def load_and_monitor_model(self, model_name: str) -> Dict[str, Any]:
        """Load a model and monitor its memory usage with peak tracking"""
        print(f"\n{'='*60}")
        print(f"🚀 Loading model: {model_name}")
        print(f"{'='*60}")
        
        # Record memory before loading
        ram_before = self.get_ram_usage()
        vram_before = self.get_vram_usage()
        
        # Start continuous memory tracking
        print("📈 Starting continuous memory monitoring...")
        self.memory_tracker.start_tracking()
        
        start_time = time.time()
        
        try:
            # Load tokenizer
            print("📦 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Add pad token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print("🤖 Loading model...")
            
            # Load model with optimizations
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_cache=True
            )
            
            # Move to device if using CPU
            if self.device.type == "cpu":
                model = model.to(self.device)
            
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f} seconds")
            
            # Stop tracking and get peak usage
            self.memory_tracker.stop_tracking()
            memory_stats = self.memory_tracker.get_memory_stats()
            
            # Record memory after loading
            ram_after = self.get_ram_usage()
            vram_after = self.get_vram_usage()
            
            # Calculate model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Memory usage calculations
            ram_used = ram_after - ram_before
            vram_used = vram_after["allocated"] - vram_before["allocated"]
            ram_delta_from_baseline = ram_after - self.baseline_ram
            vram_delta_from_baseline = vram_after["allocated"] - self.baseline_vram
            
            # Peak memory calculations
            peak_ram_delta = memory_stats.get('peak_ram_gb', ram_after) - ram_before
            peak_vram_delta = memory_stats.get('peak_vram_gb', vram_after["allocated"]) - vram_before["allocated"]
            
            # Create result dictionary
            result = {
                "model_name": model_name,
                "load_time": load_time,
                "device": str(self.device),
                "total_params": total_params,
                "trainable_params": trainable_params,
                "memory": {
                    "ram": {
                        "before_gb": ram_before,
                        "after_gb": ram_after,
                        "used_gb": ram_used,
                        "peak_during_load_gb": memory_stats.get('peak_ram_gb', ram_after),
                        "peak_delta_gb": peak_ram_delta,
                        "delta_from_baseline_gb": ram_delta_from_baseline
                    },
                    "vram": {
                        "before_gb": vram_before["allocated"],
                        "after_gb": vram_after["allocated"],
                        "used_gb": vram_used,
                        "peak_during_load_gb": memory_stats.get('peak_vram_gb', vram_after["allocated"]),
                        "peak_delta_gb": peak_vram_delta,
                        "reserved_gb": vram_after["reserved"],
                        "delta_from_baseline_gb": vram_delta_from_baseline
                    }
                },
                "memory_stats": memory_stats,
                "success": True
            }
            
            # Print detailed memory usage report
            print(f"\n📊 DETAILED MEMORY USAGE REPORT")
            print(f"├── 🧠 RAM Usage:")
            print(f"│   ├── Before Loading: {ram_before:.2f} GB")
            print(f"│   ├── After Loading: {ram_after:.2f} GB")
            print(f"│   ├── Model Usage (Final): {ram_used:.2f} GB")
            print(f"│   ├── Peak During Load: {memory_stats.get('peak_ram_gb', ram_after):.2f} GB")
            print(f"│   ├── Peak Delta: {peak_ram_delta:.2f} GB")
            print(f"│   └── Delta from Baseline: {ram_delta_from_baseline:.2f} GB")
            
            if self.device.type == "cuda":
                print(f"├── 🔥 VRAM Usage:")
                print(f"│   ├── Before Loading: {vram_before['allocated']:.2f} GB")
                print(f"│   ├── After Loading: {vram_after['allocated']:.2f} GB")
                print(f"│   ├── Model Usage (Final): {vram_used:.2f} GB")
                print(f"│   ├── Peak During Load: {memory_stats.get('peak_vram_gb', vram_after['allocated']):.2f} GB")
                print(f"│   ├── Peak Delta: {peak_vram_delta:.2f} GB")
                print(f"│   ├── Reserved (Cached): {vram_after['reserved']:.2f} GB")
                print(f"│   └── Delta from Baseline: {vram_delta_from_baseline:.2f} GB")
            
            print(f"├── 📈 Monitoring Stats:")
            print(f"│   ├── Samples Collected: {memory_stats.get('samples', 0)}")
            print(f"│   ├── Monitoring Duration: {memory_stats.get('duration', 0):.2f} seconds")
            print(f"│   ├── Average RAM: {memory_stats.get('avg_ram_gb', 0):.2f} GB")
            if self.device.type == "cuda":
                print(f"│   └── Average VRAM: {memory_stats.get('avg_vram_gb', 0):.2f} GB")
            
            print(f"├── 📈 Model Info:")
            print(f"│   ├── Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
            print(f"│   ├── Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
            print(f"│   └── Load Time: {load_time:.2f} seconds")
            
            # Now unload the model
            print(f"\n🗑️  Unloading model...")
            self.unload_model(model, tokenizer)
            
            return result
            
        except Exception as e:
            # Stop tracking on error
            self.memory_tracker.stop_tracking()
            print(f"❌ Error loading model: {str(e)}")
            return {
                "model_name": model_name,
                "error": str(e),
                "success": False
            }

    def unload_model(self, model, tokenizer):
        """Unload model and free memory with monitoring"""
        print("📈 Monitoring memory during unload...")
        
        # Start tracking unload process
        ram_before_unload = self.get_ram_usage()
        vram_before_unload = self.get_vram_usage()
        
        self.memory_tracker.start_tracking()
        
        # Delete model and tokenizer
        del model
        del tokenizer
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU cache cleared")
        
        # Stop tracking and get stats
        time.sleep(0.5)  # Allow some time for cleanup
        self.memory_tracker.stop_tracking()
        unload_stats = self.memory_tracker.get_memory_stats()
        
        print("✅ Model unloaded")
        
        # Show memory after unloading
        ram_after_unload = self.get_ram_usage()
        vram_after_unload = self.get_vram_usage()
        
        print(f"📊 Memory after unload:")
        print(f"├── RAM: {ram_after_unload:.2f} GB (freed: {ram_before_unload - ram_after_unload:.2f} GB)")
        if self.device.type == "cuda":
            vram_freed = vram_before_unload["allocated"] - vram_after_unload["allocated"]
            print(f"└── VRAM: {vram_after_unload['allocated']:.2f} GB (freed: {vram_freed:.2f} GB, Reserved: {vram_after_unload['reserved']:.2f} GB)")

    def monitor_models(self, model_names: List[str]) -> List[Dict[str, Any]]:
        """Monitor multiple models"""
        print("🔍 Model Memory Monitor Started")
        print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.setup_device()
        self.get_baseline_memory()
        
        results = []
        
        for i, model_name in enumerate(model_names, 1):
            print(f"\n🔄 Processing model {i}/{len(model_names)}")
            result = self.load_and_monitor_model(model_name)
            results.append(result)
            
            # Wait a bit between models
            if i < len(model_names):
                print("⏳ Waiting 3 seconds before next model...")
                time.sleep(3)
        
        return results

    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save results to JSON file"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "device_info": {
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
                "total_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
            },
            "baseline_memory": {
                "ram_gb": self.baseline_ram,
                "vram_gb": self.baseline_vram
            },
            "results": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Results saved to: {output_file}")

    def print_summary(self, results: List[Dict[str, Any]]):
        """Print a summary of all results with peak usage"""
        print(f"\n{'='*60}")
        print("📊 SUMMARY REPORT")
        print(f"{'='*60}")
        
        successful_models = [r for r in results if r.get("success", False)]
        failed_models = [r for r in results if not r.get("success", False)]
        
        print(f"✅ Successful: {len(successful_models)}")
        print(f"❌ Failed: {len(failed_models)}")
        
        if successful_models:
            print(f"\n📈 Memory Usage Summary (Peak Usage):")
            print(f"{'Model Name':<35} {'Peak RAM':<12} {'Peak VRAM':<12} {'Final RAM':<12} {'Final VRAM':<12} {'Params (M)':<12}")
            print("-" * 110)
            
            for result in successful_models:
                peak_ram = result["memory"]["ram"]["peak_during_load_gb"]
                peak_vram = result["memory"]["vram"]["peak_during_load_gb"] if result["device"] != "cpu" else 0
                final_ram = result["memory"]["ram"]["used_gb"]
                final_vram = result["memory"]["vram"]["used_gb"] if result["device"] != "cpu" else 0
                params = result["total_params"] / 1e6
                
                print(f"{result['model_name']:<35} {peak_ram:<12.2f} {peak_vram:<12.2f} {final_ram:<12.2f} {final_vram:<12.2f} {params:<12.1f}")
        
        if failed_models:
            print(f"\n❌ Failed Models:")
            for result in failed_models:
                print(f"  • {result['model_name']}: {result.get('error', 'Unknown error')}")

def main():
    parser = argparse.ArgumentParser(description="Model Memory Monitor with Peak Tracking")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                       help="List of model names to test")
    parser.add_argument("--output", type=str, default="model_memory_report.json",
                       help="Output file for results")
    parser.add_argument("--summary-only", action="store_true",
                       help="Only show summary, don't save detailed results")
    
    args = parser.parse_args()
    
    monitor = ModelMemoryMonitor()
    
    try:
        results = monitor.monitor_models(args.models)
        
        if not args.summary_only:
            monitor.save_results(results, args.output)
        
        monitor.print_summary(results)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()