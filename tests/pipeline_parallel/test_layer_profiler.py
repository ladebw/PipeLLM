#!/usr/bin/env python3
"""
Tests for layer profiler (Task 2.1).

WARNING: SIMULATED DATA — not from real hardware
These tests verify the profiling infrastructure, not actual performance.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import layer profiler
from src.pipeline_parallel.layer_profiler import (
    LayerTiming,
    ExecutionTimeline,
    LayerProfiler
)


class TestLayerTiming:
    """Tests for LayerTiming dataclass."""
    
    def test_creation(self):
        """Test creating a LayerTiming instance."""
        timing = LayerTiming(
            layer_id=1,
            layer_name="attention",
            compute_time_ms=10.5,
            memory_time_ms=3.2,
            total_time_ms=13.7,
            compute_percentage=76.6,
            memory_percentage=23.4,
            bottleneck="compute",
            notes="Test layer"
        )
        
        assert timing.layer_id == 1
        assert timing.layer_name == "attention"
        assert timing.compute_time_ms == 10.5
        assert timing.memory_time_ms == 3.2
        assert timing.total_time_ms == 13.7
        assert timing.compute_percentage == 76.6
        assert timing.memory_percentage == 23.4
        assert timing.bottleneck == "compute"
        assert timing.notes == "Test layer"
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        timing = LayerTiming(
            layer_id=1,
            layer_name="attention",
            compute_time_ms=10.5,
            memory_time_ms=3.2,
            total_time_ms=13.7,
            compute_percentage=76.6,
            memory_percentage=23.4,
            bottleneck="compute",
            notes="Test layer"
        )
        
        result = timing.to_dict()
        
        assert isinstance(result, dict)
        assert result["layer_id"] == 1
        assert result["layer_name"] == "attention"
        assert result["compute_time_ms"] == 10.5
        assert result["memory_time_ms"] == 3.2
        assert result["total_time_ms"] == 13.7
        assert result["compute_percentage"] == 76.6
        assert result["memory_percentage"] == 23.4
        assert result["bottleneck"] == "compute"
        assert result["notes"] == "Test layer"
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "layer_id": 2,
            "layer_name": "ffn",
            "compute_time_ms": 8.2,
            "memory_time_ms": 4.1,
            "total_time_ms": 12.3,
            "compute_percentage": 66.7,
            "memory_percentage": 33.3,
            "bottleneck": "memory",
            "notes": "FFN layer"
        }
        
        timing = LayerTiming.from_dict(data)
        
        assert timing.layer_id == 2
        assert timing.layer_name == "ffn"
        assert timing.compute_time_ms == 8.2
        assert timing.memory_time_ms == 4.1
        assert timing.total_time_ms == 12.3
        assert timing.compute_percentage == 66.7
        assert timing.memory_percentage == 33.3
        assert timing.bottleneck == "memory"
        assert timing.notes == "FFN layer"
    
    def test_bottleneck_detection(self):
        """Test bottleneck detection logic."""
        # Compute bottleneck
        compute_timing = LayerTiming(
            layer_id=1,
            layer_name="test",
            compute_time_ms=15.0,
            memory_time_ms=5.0,
            total_time_ms=20.0,
            compute_percentage=75.0,
            memory_percentage=25.0,
            bottleneck="",  # Will be auto-detected
            notes=""
        )
        
        # Memory bottleneck
        memory_timing = LayerTiming(
            layer_id=2,
            layer_name="test",
            compute_time_ms=5.0,
            memory_time_ms=15.0,
            total_time_ms=20.0,
            compute_percentage=25.0,
            memory_percentage=75.0,
            bottleneck="",  # Will be auto-detected
            notes=""
        )
        
        # Balanced
        balanced_timing = LayerTiming(
            layer_id=3,
            layer_name="test",
            compute_time_ms=10.0,
            memory_time_ms=10.0,
            total_time_ms=20.0,
            compute_percentage=50.0,
            memory_percentage=50.0,
            bottleneck="",  # Will be auto-detected
            notes=""
        )
        
        # Bottleneck should be auto-detected
        assert compute_timing.bottleneck == "compute"
        assert memory_timing.bottleneck == "memory"
        assert balanced_timing.bottleneck == "balanced"


class TestExecutionTimeline:
    """Tests for ExecutionTimeline dataclass."""
    
    def test_creation(self):
        """Test creating an ExecutionTimeline instance."""
        layer_timings = [
            LayerTiming(
                layer_id=1,
                layer_name="layer1",
                compute_time_ms=10.0,
                memory_time_ms=5.0,
                total_time_ms=15.0,
                compute_percentage=66.7,
                memory_percentage=33.3,
                bottleneck="compute",
                notes=""
            ),
            LayerTiming(
                layer_id=2,
                layer_name="layer2",
                compute_time_ms=8.0,
                memory_time_ms=12.0,
                total_time_ms=20.0,
                compute_percentage=40.0,
                memory_percentage=60.0,
                bottleneck="memory",
                notes=""
            )
        ]
        
        timeline = ExecutionTimeline(
            model_name="test_model",
            sequence_length=512,
            batch_size=1,
            num_layers=2,
            layer_timings=layer_timings,
            total_compute_time_ms=18.0,
            total_memory_time_ms=17.0,
            total_time_ms=35.0,
            compute_percentage=51.4,
            memory_percentage=48.6,
            overall_bottleneck="balanced",
            overlap_potential_pct=85.0,
            notes="Test timeline"
        )
        
        assert timeline.model_name == "test_model"
        assert timeline.sequence_length == 512
        assert timeline.batch_size == 1
        assert timeline.num_layers == 2
        assert len(timeline.layer_timings) == 2
        assert timeline.total_compute_time_ms == 18.0
        assert timeline.total_memory_time_ms == 17.0
        assert timeline.total_time_ms == 35.0
        assert timeline.compute_percentage == 51.4
        assert timeline.memory_percentage == 48.6
        assert timeline.overall_bottleneck == "balanced"
        assert timeline.overlap_potential_pct == 85.0
        assert timeline.notes == "Test timeline"
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        layer_timings = [
            LayerTiming(
                layer_id=1,
                layer_name="layer1",
                compute_time_ms=10.0,
                memory_time_ms=5.0,
                total_time_ms=15.0,
                compute_percentage=66.7,
                memory_percentage=33.3,
                bottleneck="compute",
                notes=""
            )
        ]
        
        timeline = ExecutionTimeline(
            model_name="test_model",
            sequence_length=512,
            batch_size=1,
            num_layers=1,
            layer_timings=layer_timings,
            total_compute_time_ms=10.0,
            total_memory_time_ms=5.0,
            total_time_ms=15.0,
            compute_percentage=66.7,
            memory_percentage=33.3,
            overall_bottleneck="compute",
            overlap_potential_pct=80.0,
            notes="Test"
        )
        
        result = timeline.to_dict()
        
        assert isinstance(result, dict)
        assert result["model_name"] == "test_model"
        assert result["sequence_length"] == 512
        assert result["num_layers"] == 1
        assert "layer_timings" in result
        assert len(result["layer_timings"]) == 1
        assert result["total_time_ms"] == 15.0
        assert result["overlap_potential_pct"] == 80.0
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "model_name": "test_model",
            "sequence_length": 1024,
            "batch_size": 1,
            "num_layers": 2,
            "layer_timings": [
                {
                    "layer_id": 1,
                    "layer_name": "layer1",
                    "compute_time_ms": 12.0,
                    "memory_time_ms": 6.0,
                    "total_time_ms": 18.0,
                    "compute_percentage": 66.7,
                    "memory_percentage": 33.3,
                    "bottleneck": "compute",
                    "notes": ""
                }
            ],
            "total_compute_time_ms": 12.0,
            "total_memory_time_ms": 6.0,
            "total_time_ms": 18.0,
            "compute_percentage": 66.7,
            "memory_percentage": 33.3,
            "overall_bottleneck": "compute",
            "overlap_potential_pct": 75.0,
            "notes": "From dict"
        }
        
        timeline = ExecutionTimeline.from_dict(data)
        
        assert timeline.model_name == "test_model"
        assert timeline.sequence_length == 1024
        assert timeline.num_layers == 2
        assert len(timeline.layer_timings) == 1
        assert timeline.layer_timings[0].layer_id == 1
        assert timeline.total_time_ms == 18.0
        assert timeline.overlap_potential_pct == 75.0
    
    def test_calculate_statistics(self):
        """Test statistics calculation."""
        layer_timings = [
            LayerTiming(
                layer_id=1,
                layer_name="layer1",
                compute_time_ms=10.0,
                memory_time_ms=5.0,
                total_time_ms=15.0,
                compute_percentage=66.7,
                memory_percentage=33.3,
                bottleneck="compute",
                notes=""
            ),
            LayerTiming(
                layer_id=2,
                layer_name="layer2",
                compute_time_ms=8.0,
                memory_time_ms=12.0,
                total_time_ms=20.0,
                compute_percentage=40.0,
                memory_percentage=60.0,
                bottleneck="memory",
                notes=""
            )
        ]
        
        timeline = ExecutionTimeline(
            model_name="test_model",
            sequence_length=512,
            batch_size=1,
            num_layers=2,
            layer_timings=layer_timings,
            total_compute_time_ms=18.0,
            total_memory_time_ms=17.0,
            total_time_ms=35.0,
            compute_percentage=51.4,
            memory_percentage=48.6,
            overall_bottleneck="balanced",
            overlap_potential_pct=85.0,
            notes="Test"
        )
        
        stats = timeline.calculate_statistics()
        
        assert isinstance(stats, dict)
        assert "total_layers" in stats
        assert "total_time_ms" in stats
        assert "compute_dominant_layers" in stats
        assert "memory_dominant_layers" in stats
        assert "balanced_layers" in stats
        assert "avg_compute_percentage" in stats
        assert "avg_memory_percentage" in stats
        
        assert stats["total_layers"] == 2
        assert stats["total_time_ms"] == 35.0
        assert stats["compute_dominant_layers"] == 1
        assert stats["memory_dominant_layers"] == 1
        assert stats["balanced_layers"] == 0
        assert stats["avg_compute_percentage"] == 53.35  # (66.7 + 40.0) / 2
        assert stats["avg_memory_percentage"] == 46.65   # (33.3 + 60.0) / 2


class TestLayerProfiler:
    """Tests for LayerProfiler class."""
    
    def test_initialization(self, tmp_path):
        """Test profiler initialization."""
        profiler = LayerProfiler(output_dir=tmp_path)
        
        assert profiler.output_dir == tmp_path
        assert profiler.output_dir.exists()
        assert hasattr(profiler, 'system_info')
        assert hasattr(profiler, 'configs')
        
        # Check system info
        assert "system" in profiler.system_info
        assert "python_version" in profiler.system_info
        assert "cuda_available" in profiler.system_info
        assert profiler.system_info["cuda_available"] is False  # Simulation
    
    def test_get_system_info(self):
        """Test system info collection."""
        profiler = LayerProfiler()
        system_info = profiler._get_system_info()
        
        assert isinstance(system_info, dict)
        assert "system" in system_info
        assert "python_version" in system_info
        assert "timestamp" in system_info
        assert "cuda_available" in system_info
        assert "note" in system_info
        assert "WARNING" in system_info["note"]  # Should contain warning
    
    def test_create_mock_transformer_layer(self):
        """Test mock transformer layer creation."""
        profiler = LayerProfiler()
        
        layer = profiler._create_mock_transformer_layer(
            hidden_size=4096,
            compute_time_ms=10.0,
            memory_time_ms=5.0
        )
        
        assert isinstance(layer, dict)
        assert "hidden_size" in layer
        assert "compute_time_ms" in layer
        assert "memory_time_ms" in layer
        assert "total_time_ms" in layer
        assert layer["hidden_size"] == 4096
        assert layer["compute_time_ms"] == 10.0
        assert layer["memory_time_ms"] == 5.0
        assert layer["total_time_ms"] == 15.0
    
    def test_create_mock_model(self):
        """Test mock model creation."""
        profiler = LayerProfiler()
        
        model = profiler._create_mock_model(
            num_layers=4,
            hidden_size=4096,
            base_compute_time=10.0,
            base_memory_time=5.0
        )
        
        assert isinstance(model, dict)
        assert "num_layers" in model
        assert "hidden_size" in model
        assert "layers" in model
        assert model["num_layers"] == 4
        assert model["hidden_size"] == 4096
        assert len(model["layers"]) == 4
        
        # Check layer structure
        for i, layer in enumerate(model["layers"]):
            assert "layer_id" in layer
            assert "layer_name" in layer
            assert "compute_time_ms" in layer
            assert "memory_time_ms" in layer
            assert layer["layer_id"] == i + 1
            assert f"layer_{i+1}" in layer["layer_name"]
    
    def test_simulate_layer_execution(self):
        """Test layer execution simulation."""
        profiler = LayerProfiler()
        
        # Create a mock layer
        mock_layer = {
            "layer_id": 1,
            "layer_name": "attention",
            "compute_time_ms": 12.5,
            "memory_time_ms": 7.5,
            "total_time_ms": 20.0
        }
        
        timing = profiler._simulate_layer_execution(mock_layer)
        
        assert isinstance(timing, LayerTiming)
        assert timing.layer_id == 1
        assert timing.layer_name == "attention"
        assert timing.compute_time_ms == 12.5
        assert timing.memory_time_ms == 7.5
        assert timing.total_time_ms == 20.0
        assert timing.compute_percentage == 62.5  # 12.5 / 20.0 * 100
        assert timing.memory_percentage == 37.5   # 7.5 / 20.0 * 100
        assert timing.bottleneck == "compute"  # 62.5% > 37.5%
    
    def test_calculate_overlap_potential(self):
        """Test overlap potential calculation."""
        profiler = LayerProfiler()
        
        # Test case 1: Memory-bound (high overlap potential)
        memory_timings = [
            LayerTiming(
                layer_id=1,
                layer_name="layer1",
                compute_time_ms=5.0,
                memory_time_ms=15.0,
                total_time_ms=20.0,
                compute_percentage=25.0,
                memory_percentage=75.0,
                bottleneck="memory",
                notes=""
            ),
            LayerTiming(
                layer_id=2,
                layer_name="layer2",
                compute_time_ms=6.0,
                memory_time_ms=14.0,
                total_time_ms=20.0,
                compute_percentage=30.0,
                memory_percentage=70.0,
                bottleneck="memory",
                notes=""
            )
        ]
        
        overlap1 = profiler._calculate_overlap_potential(memory_timings)
        assert overlap1 > 90.0  # High overlap potential for memory-bound
        
        # Test case 2: Compute-bound (lower overlap potential)
        compute_timings = [
            LayerTiming(
                layer_id=1,
                layer_name="layer1",
                compute_time_ms=15.0,
                memory_time_ms=5.0,
                total_time_ms=20.0,
                compute_percentage=75.0,
                memory_percentage=25.0,
                bottleneck="compute",
                notes=""
            ),
            LayerTiming(
                layer_id=2,
                layer_name="layer2",
                compute_time_ms=14.0,
                memory_time_ms=6.0,
                total_time_ms=20.0,
                compute_percentage=70.0,
                memory_percentage=30.0,
                bottleneck="compute",
                notes=""
            )
        ]
        
        overlap2 = profiler._calculate_overlap_potential(compute_timings)
        assert overlap2 < 50.0  # Lower overlap potential for compute-bound
    
    def test_profile_model(self):
        """Test model profiling."""
        profiler = LayerProfiler()
        
        timeline = profiler.profile_model(
            model_name="test_model",
            sequence_length=512,
            batch_size=1,
            num_layers=4,
            hidden_size=4096
        )
        
        assert isinstance(timeline, ExecutionTimeline)
        assert timeline.model_name == "test_model"
        assert timeline.sequence_length == 512
        assert timeline.batch_size == 1
        assert timeline.num_layers == 4
        
        # Should have timings for all layers
        assert len(timeline.layer_timings) == 4
        
        # Check layer timings
        for i, timing in enumerate(timeline.layer_timings):
            assert timing.layer_id == i + 1
            assert f"layer_{i+1}" in timing.layer_name
            assert timing.compute_time_ms > 0
            assert timing.memory_time_ms > 0
            assert timing.total_time_ms > 0
            assert 0 <= timing.compute_percentage <= 100
            assert 0 <= timing.memory_percentage <= 100
            assert timing.bottleneck in ["compute", "memory", "balanced"]
        
        # Check overall statistics
        assert timeline.total_compute_time_ms > 0
        assert timeline.total_memory_time_ms > 0
        assert timeline.total_time_ms > 0
        assert 0 <= timeline.compute_percentage <= 100
        assert 0 <= timeline.memory_percentage <= 100
        assert timeline.overall_bottleneck in ["compute", "memory", "balanced"]
        assert 0 <= timeline.overlap_potential_pct <= 100
    
    def test_profile_all_configurations(self):
        """Test profiling all configurations."""
        profiler = LayerProfiler()
        
        results = profiler.profile_all_configurations()
        
        # Should have results for all 4 configurations
        assert len(results) == 4
        assert "small_model" in results
        assert "medium_model" in results
        assert "large_model" in results
        assert "deep_model" in results
        
        # All results should be ExecutionTimeline instances
        for timeline in results.values():
            assert isinstance(timeline, ExecutionTimeline)
            
            # Check basic properties
            assert timeline.model_name in ["small_model", "medium_model", "large_model", "deep_model"]
            assert timeline.num_layers > 0
            assert len(timeline.layer_timings) == timeline.num_layers
    
    def test_save_results(self, tmp_path):
        """Test saving profiling results."""
        profiler = LayerProfiler(output_dir=tmp_path)
        
        # Create a mock timeline
        layer_timings = [
            LayerTiming(
                layer_id=1,
                layer_name="layer1",
                compute_time_ms=10.0,
                memory_time_ms=5.0,
                total_time_ms=15.0,
                compute_percentage=66.7,
                memory_percentage=33.3,
                bottleneck="compute",
                notes=""
            )
        ]
        
        timeline = ExecutionTimeline(
            model_name="test_model",
            sequence_length=512,
            batch_size=1,
            num_layers=1,
            layer_timings=layer_timings,
            total_compute_time_ms=10.0,
            total_memory_time_ms=5.0,
            total_time_ms=15.0,
            compute_percentage=66.7,
            memory_percentage=33.3,
            overall_bottleneck="compute",
            overlap_potential_pct=80.0,
            notes="Test"
        )
        
        results = {"test_model": timeline}
        
        # Save results
        output_path = profiler.save_results(results)
        
        assert output_path.exists()
        
        # Load and verify
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert "test_model" in saved_data
        assert "_metadata" in saved_data
        assert saved_data["_metadata"]["task"] == "2.1"
        assert "SIMULATED DATA" in saved_data["_metadata"]["warning"]
        
        model_data = saved_data["test_model"]
        assert model_data["model_name"] == "test_model"
        assert model_data["sequence_length"] == 512
        assert model_data["overlap_potential_pct"] == 80.0
    
    def test_generate_summary_report(self, tmp_path):
        """Test generating summary report."""
        profiler = LayerProfiler(output_dir=tmp_path)
        
        # Create mock results
        layer_timings = [
            LayerTiming(
                layer_id=1,
                layer_name="layer1",
                compute_time_ms=10.0,
                memory_time_ms=5.0,
                total_time_ms=15.0,
                compute_percentage=66.7,
                memory_percentage=33.3,
                bottleneck="compute",
                notes=""
            )
        ]
        
        timeline = ExecutionTimeline(
            model_name="small_model",
            sequence_length=512,
            batch_size=1,
            num_layers=1,
            layer_timings=layer_timings,
            total_compute_time_ms=10.0,
            total_memory_time_ms=5.0,
            total_time_ms=15.0,
            compute_percentage=66.7,
            memory_percentage=33.3,
            overall_bottleneck="compute",
            overlap_potential_pct=80.0,
            notes="Test"
        )
        
        results = {"small_model": timeline}
        
        # Generate report
        report = profiler.generate_summary_report(results)
        
        assert isinstance(report, str)
        assert "LAYER PROFILING REPORT" in report
        assert "Task 2.1" in report
        assert "WARNING" in report
        assert "SIMULATED" in report
        assert "small_model" in report
        assert "512" in report
        assert "80.0%" in report  # Overlap potential
    
    def test_save_summary_report(self, tmp_path):
        """Test saving summary report."""
        profiler = LayerProfiler(output_dir=tmp_path)
        
        # Create mock results
        layer_timings = [
            LayerTiming(
                layer_id=1,
                layer_name="layer1",
                compute_time_ms=10.0,
                memory_time_ms=5.0,
                total_time_ms=15.0,
                compute_percentage=66.7,
                memory_percentage=33.3,
                bottleneck="compute",
                notes=""
            )
        ]
        
        timeline = ExecutionTimeline(
            model_name="test_model",
            sequence_length=512,
            batch_size=1,
            num_layers=1,
            layer_timings=layer_timings,
            total_compute_time_ms=10.0,
            total_memory_time_ms=5.0,
            total_time_ms=15.0,
            compute_percentage=66.7,
            memory_percentage=33.3,
            overall_bottleneck="compute",
            overlap_potential_pct=80.0,
            notes="Test"
        )
        
        results = {"test_model": timeline}
        
        # Save report
        output_path = profiler.save_summary_report(results)
        
        assert output_path.exists()
        
        # Load and verify
        with open(output_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        assert "LAYER PROFILING REPORT" in report_content
        assert "WARNING" in report_content
        assert "SIMULATED" in report_content


class TestIntegration:
    """Integration tests for layer profiler."""
    
    def test_full_profiling_flow(self, tmp_path):
        """Test full profiling flow from initialization to result saving."""
        # Initialize profiler
        profiler = LayerProfiler(output_dir=tmp_path)
        
        # Profile single model
        timeline = profiler.profile_model(
            model_name="integration_test",
            sequence_length=512,
            batch_size=1,
            num_layers=2
        )
        
        # Verify result
        assert isinstance(timeline, ExecutionTimeline)
        assert timeline.model_name == "integration_test"
        assert timeline.sequence_length == 512
        assert timeline.num_layers == 2
        
        # Save results
        results = {"integration_test": timeline}
        json_path = profiler.save_results(results)
        report_path = profiler.save_summary_report(results)
        
        # Verify files were created
        assert json_path.exists()
        assert report_path.exists()
        
        # Verify JSON content
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert "integration_test" in json_data
        assert "_metadata" in json_data
        
        # Verify report content
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = f.read()
        
        assert "integration_test" in report_data
        assert "512" in report_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])