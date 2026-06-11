# TIØ4900 Replication Repository

This repository contains the replication code for our TIØ4900 master's thesis on machine learning evidence in U.S. bond excess return forecasting: Machine Learning Evidence on Bond Risk Premia and the Spanning Hypothesis.

The codebase is organized around Jupyter notebooks and supporting Python modules for constructing predictors, running expanding-window forecasts, evaluating models, and producing the tables and figures used in the thesis.

## Main entry point

The main playground and entry point is:

```text
notebooks/init.ipynb
```

Start there to inspect the data pipeline, construct yield and macro predictors, generate excess returns, and run smaller exploratory workflows.

## Main result generation

The main forecasting results are produced from the orchestrator notebooks in:

```text
notebooks/orchestrators/
```

The most important orchestrators are:

```text
notebooks/orchestrators/orchestrator_ann_runs.ipynb
notebooks/orchestrators/orchestrator_linear_runs.ipynb
notebooks/orchestrators/orchestrator_tree_runs.ipynb
```

These notebooks run the main model families used in the thesis:

- ANN models and ensembles are generated from the ANN orchestrator notebooks.
- Linear model results are generated from the linear orchestrator notebooks.
- Tree-based model results are generated from the tree orchestrator notebooks.

The ANN tables reported in the text are obtained from the `CSSED_ANN` and `monthly_results` notebooks. The linear and tree model tables are obtained directly from the corresponding orchestrator notebooks.

## Repository structure

```text
data/
models/
notebooks/
utils/
```

### `data/`

Contains the CSV files used by the replication code. For the data provenance, see:

```text
data/readme.txt
```

No additional data-source documentation is provided here; the `data/readme.txt` file is the authoritative reference for the included datasets.

### `models/`

Contains model implementations and model configuration code used by the forecasting notebooks. This includes wrappers and configurations for linear models, tree-based models, and artificial neural networks.

### `notebooks/`

Contains the main analysis notebooks. The most important notebook for getting started is `init.ipynb`, while the main forecasting runs are handled by the notebooks under `notebooks/orchestrators/`.

Additional notebooks are used for result aggregation, monthly results, CSSED figures, and other thesis-specific reporting workflows.

### `utils/`

Contains shared helper code for data loading, return construction, forward-rate construction, expanding-window forecasting, result persistence, plotting, model orchestration, macro grouping, and SHAP-related analysis.

## Typical workflow

1. Check that the required CSV files are present in `data/`.
2. Open `notebooks/init.ipynb` to inspect the main data and forecasting setup.
3. Run the relevant orchestrator notebook under `notebooks/orchestrators/`.
4. Use the result and visualization notebooks to reproduce the reported tables and figures.

## Data disclaimer

The included data CSV files are not owned by us. They are included only to make the replication workflow easier to run. If any rights holder or data provider requests that a dataset be removed, it will be taken down.

This README was written by ChatGPT. 
