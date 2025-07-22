# tests/test_evaluate.py
import os
import sys
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import all functions from the source file
from src.evaluate import (calculate_pinball_loss, comprehensive_evaluation,
                          coverage_analysis, plot_coverage_bar,
                          plot_error_distribution,
                          plot_prediction_interval_width,
                          plot_prediction_intervals, plot_training_history,
                          scatter_pred_vs_true)

# --- Test Core Logic Functions ---


def test_calculate_pinball_loss():
    """Test the pinball loss calculation with known values."""
    y_true = np.array([10, 10, 10, 10])
    y_pred = np.array([5, 15, 10, 5])  # errors: 5, -5, 0, 5
    quantile = 0.8

    # Expected calculation:
    # err = [5, -5, 0, 5]
    # positive_err = quantile * err = [4, -4, 0, 4]
    # negative_err = (quantile - 1) * err = [-1, 1, 0, -1]
    # max = [4, 1, 0, 4]
    # mean = (4 + 1 + 0 + 4) / 4 = 9 / 4 = 2.25
    expected_loss = 2.25

    assert calculate_pinball_loss(y_true, y_pred, quantile) == pytest.approx(
        expected_loss
    )

    # Test another quantile
    quantile_2 = 0.2
    # Expected calculation:
    # err = [5, -5, 0, 5]
    # positive_err = quantile * err = [1, -1, 0, 1]
    # negative_err = (quantile - 1) * err = [-4, 4, 0, -4]
    # max = [1, 4, 0, 1]
    # mean = (1 + 4 + 0 + 1) / 4 = 6 / 4 = 1.5
    expected_loss_2 = 1.5
    assert calculate_pinball_loss(y_true, y_pred, quantile_2) == pytest.approx(
        expected_loss_2
    )


# --- Test Plotting Functions ---
# For plotting functions, the primary test is to ensure they run without errors given valid inputs.
# We use `plt.close('all')` to prevent figures from being displayed during tests.


@patch("matplotlib.pyplot.show")
def test_plotting_functions_run_without_error(
    mock_show, sample_y_true, sample_predictions, sample_history_dict
):
    """
    Tests that all plotting functions execute without raising an error.
    `patch` is used to prevent plot windows from opening during tests.
    """
    try:
        # Flatten predictions for functions that expect 1D arrays
        flat_preds = {q: p.flatten() for q, p in sample_predictions.items()}
        y_true_flat = sample_y_true.flatten()
        median_pred = flat_preds[0.5]

        # Test each plotting function
        plot_prediction_intervals(y_true_flat, flat_preds)
        coverage_analysis(y_true_flat, flat_preds)
        plot_training_history(sample_history_dict, "TestModel")
        scatter_pred_vs_true(y_true_flat, median_pred, 0.5)
        plot_prediction_interval_width(flat_preds)
        plot_error_distribution(y_true_flat, median_pred)
        plot_coverage_bar(y_true_flat, flat_preds)

    except Exception as e:
        pytest.fail(f"A plotting function failed to execute: {e}")
    finally:
        plt.close("all")  # Clean up any figures created


# --- Test the Main Evaluation Function ---


@patch("src.evaluate.display")  # Mock the IPython display function
@patch("matplotlib.pyplot.show")  # Mock pyplot.show
def test_comprehensive_evaluation_runs_successfully(
    mock_show,
    mock_display,
    capsys,
    sample_y_true,
    sample_predictions,
    sample_history_dict,
):
    """
    Test the main `comprehensive_evaluation` function to ensure it runs end-to-end.
    - `capsys` fixture captures print output.
    - Mocks are used for `display` and `show`.
    """
    model_name = "Test_CNN_Model"
    training_time = 123.456

    # Reshape y_true to match typical model output shape (e.g., [n_samples, 1])
    y_test_unscaled = sample_y_true.reshape(-1, 1)

    # Run the comprehensive evaluation
    comprehensive_evaluation(
        predictions=sample_predictions,
        y_test_unscaled=y_test_unscaled,
        history_dict=sample_history_dict,
        model_name=model_name,
        training_time=training_time,
    )

    # 1. Check if the function printed its title
    captured = capsys.readouterr()
    assert f"COMPREHENSIVE EVALUATION FOR MODEL: {model_name}" in captured.out
    assert "Performance Metrics Table" in captured.out

    # 2. Check if the display function was called with a DataFrame
    mock_display.assert_called_once()
    call_args = mock_display.call_args[0]
    assert isinstance(
        call_args[0], pd.io.formats.style.Styler
    )  # Check if a styled DataFrame was passed

    # 3. Check if the plots were generated (mock_show was called)
    # The number of calls depends on the number of plotting functions inside
    assert mock_show.call_count > 0

    plt.close("all")


@patch("src.evaluate.display")
@patch("matplotlib.pyplot.show")
def test_comprehensive_evaluation_no_median(
    mock_show,
    mock_display,
    capsys,
    sample_y_true,
    sample_predictions_no_median,
    sample_history_dict,
):
    """
    Test the evaluation function handles cases where no median (q=0.5) prediction is present.
    """
    model_name = "No_Median_Model"
    y_test_unscaled = sample_y_true.reshape(-1, 1)

    comprehensive_evaluation(
        predictions=sample_predictions_no_median,
        y_test_unscaled=y_test_unscaled,
        history_dict=sample_history_dict,
        model_name=model_name,
        training_time=None,  # Test with no training time
    )

    captured = capsys.readouterr()

    # Check that metrics dependent on the median are handled gracefully
    # Here, we expect the DataFrame to be created with NaN values for MAE/RMSE
    mock_display.assert_called_once()
    styler_obj = mock_display.call_args[0][0]
    df = styler_obj.data

    assert pd.isna(df["MAE (q=0.5)"].iloc[0])
    assert pd.isna(df["RMSE (q=0.5)"].iloc[0])
    assert "N/A" in df["Total Training Time (s)"].iloc[0]

    # Check that scatter plot vs actual is NOT generated
    # Total plots = history, interval_width, intervals, coverage. Scatter plot is skipped.
    assert "Predicted vs Actual (for Median):" not in captured.out

    plt.close("all")


if __name__ == "__main__":
    # This allows the test script to be run directly
    pytest.main()
