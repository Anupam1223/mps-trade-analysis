# MPS Stock Market Predictor

This project explores the application of a **Matrix Product State (MPS)**, a type of tensor network model inspired by quantum physics, for predicting directional movements in the stock market. The goal is to determine if these advanced models can capture temporal correlations in financial time-series data more effectively than traditional models.

The pipeline ingests historical stock data from the [Alpaca Markets API](https://alpaca.markets/), preprocesses it to create relevant features, and then trains an MPS-based neural network built with TensorFlow and TensorNetwork to predict whether a stock's price will rise or fall.

---

## Key Features

* **Data Ingestion**: Fetches daily historical stock data for multiple symbols using the Alpaca API.
* **Feature Engineering**: Creates a rich feature set including price returns, log volume, and various technical indicators (RSI, MACD, EMA ratios).
* **Matrix Product State Model**: Implements a custom Keras layer for an MPS model to process sequential data.
* **Stable Training**: Incorporates `LayerNormalization` and `Gradient Clipping` to ensure stable training for the deep tensor network architecture.
* **Evaluation Suite**: Provides detailed model evaluation, including loss/accuracy plots, a confusion matrix, and an ROC curve.

---

## Technology Stack

* **Python 3.9+**
* **TensorFlow**: For building and training the neural network.
* **TensorNetwork**: For creating the Matrix Product State model.
* **Alpaca Trade API**: For historical stock data.
* **Pandas**: For data manipulation and analysis.
* **scikit-learn**: For data scaling and evaluation metrics.
* **TA**: For generating technical analysis indicators.
* **Matplotlib**: For plotting and visualization.

---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd mps-trade-analysis
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install tensorflow tensornetwork alpaca-trade-api pandas scikit-learn ta matplotlib seaborn
    ```
    *(Note: You can also generate a `requirements.txt` file from your environment using `pip freeze > requirements.txt`)*

4.  **Set up API Keys:**
    This project requires API keys from Alpaca. You need to set them as environment variables.

    **On macOS/Linux:**
    ```bash
    export APCA_API_KEY_ID='YOUR_API_KEY'
    export APCA_API_SECRET_KEY='YOUR_SECRET_KEY'
    ```

    **On Windows (Command Prompt):**
    ```bash
    set APCA_API_KEY_ID="YOUR_API_KEY"
    set APCA_API_SECRET_KEY="YOUR_SECRET_KEY"
    ```

---

## Usage

To run the entire pipeline from data ingestion to model evaluation, simply execute the main script:

```bash
python main.py