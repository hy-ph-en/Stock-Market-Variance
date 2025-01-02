# Market Volatility Impact Analysis

This project investigates whether increased volatility in training data can make machine learning models more robust for market prediction. It implements and compares Random Forest and LSTM models using cryptocurrency price data.

## Key Features

- Random Forest and LSTM models for price prediction
- Technical indicator calculations (RSI, MACD, Stochastic Oscillator, etc.)
- Volatility analysis tools
- Performance evaluation metrics
- Trading simulation capabilities

## Dependencies

```
numpy
pandas
matplotlib
scikit-learn
tensorflow
requests
math
```

## Project Structure

- `Random_Forest_Model.py`: Implementation of Random Forest predictor
- `Data_Organiser.py`: Data preprocessing and technical indicator calculations
- `Unit_Creation.py`: Technical indicator implementations (RSI, MACD, etc.)

## Technical Indicators

The project calculates several technical indicators:
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Stochastic Oscillator
- Williams Percentage Range
- Rate of Change
- Balance Volume

## Usage

1. Data preparation:
```python
from Data_Organiser import Data_Random_Forest
# Prepare training and testing datasets
training, testing = Data_Random_Forest(data)
```

2. Model training and prediction:
```python
from Random_Forest_Model import RandomForest
predictions = RandomForest(n_estimators=100, 
                         oob_score=True,
                         prediction_days=5,
                         criterion='mse',
                         min_samples_leaf=1,
                         training=training,
                         testing=testing)
```

3. Performance evaluation:
```python
from Purchase_Agent import Purchase_Agent
# Evaluate trading performance
returns = Purchase_Agent(predictions, actual_data)
```

## Model Parameters

### Random Forest
- n_estimators: Number of trees
- min_samples_leaf: Minimum samples per leaf
- max_depth: Maximum tree depth (default 60)
- prediction_days: Number of days to predict ahead

## Results Analysis

The project analyzes model performance across different volatility levels:
- Low volatility (variance < 0.3)
- Medium volatility (variance 0.3-0.5)
- High volatility (variance > 0.5)

Performance metrics include:
- R² Score
- Mean Absolute Error
- Mean Squared Error
- Root Mean Squared Error
- Classification accuracy for price movement direction
