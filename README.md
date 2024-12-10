# DLWP-eval-extremes
The repository documents code for _"Validating Deep-Learning Weather Forecast Models on Recent High-Impact Extreme Events"_ by  Olivier C. Pasche, Jonathan Wider, Zhongwei Zhang, Jakob Zscheischler, and Sebastian Engelke [[link](https://doi.org/10.1175/AIES-D-24-0033.1)]. We focus on the analyses conducted for our case studies and don't provide details on how the AI models were run - we built on the code released by the respective modeling groups, see section "Prediction models".

## Caveats
This preliminary version lacks the code and data to produce Figures 3 and A1 in the paper. We will create a complete and tagged version in the near future.

## Setup
We provide two environment files: `eval_env.yml` is the file we used to create the environment, and `eval_log.yml` was made with `conda env export` to provide the exact version we used.

This should enable recreating our environment through `conda env create -f <environment-name>.yml`.

## Data
We release preprocessed ground truth and prediction data for the three case studies considered in the paper as a [zenodo dataset](). The dataset is created using several sources:
- for the ground truth data sets, we use data released through WeatherBench 2 ([[paper](https://doi.org/10.1029/2023MS004019)] [[dataset documentation](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)]) when possible. In particular, we use their ERA5 climatology in the Pacific Northwest heatwave case study.
- when the ground truth data is not available through WeatherBench 2, we download it from ECMWF.
- We ran all ML forecasts ourselves, using ERA5 to initialize the models.
- HRES forecasts were retrieved from the ECMWF operational archive and the TIGGE data retrieval portal.

For details, including the license statements of the utilized ECMWF data sets, see the documentation of the [zenodo dataset]().

## Prediction models
We compare the following AI weather prediction models:
- GraphCast [[paper](https://doi.org/10.1126/science.adi2336)] [[GitHub](https://github.com/google-deepmind/graphcast)]
- PanguWeather [[paper](https://doi.org/10.1038/s41586-023-06185-3)] [[GitHub](https://github.com/198808xc/Pangu-Weather)]
- FourCastNet [[paper](https://doi.org/10.48550/arXiv.2202.11214)] [[GitHub]([https://github.com/NVlabs/FourCastNet)]

For GraphCast, we use the version released on GitHub, not to be confused with  'GraphCast_small' and 'GraphCast_operational'. For PanguWeather we used a combination of the authors' 6h and 24h models, following the "hierarchical temporal aggregation strategy" they developed.

## Shapefiles
For the case study on the 2023 South Asia humid heatwave, we used the shapefiles contained in the `Shapefiles` subdirectory to mask a region that is comparable to the study _"Extreme humid heat in South Asia in April 2023, largely driven by climate change, detrimental to vulnerable and disadvantaged communities"_ (2023), [[paper](https://doi.org/10.25561/104092)]. We use country boundaries from the “World Administrative Boundaries - Countries and Territories” ([[link](https://public.opendatasoft.com/explore/dataset/world-administrative-boundaries/information/)], Open Government License 3.0) data set by the World Food Programme and for the India-Bangladesh region we additionally use the (1976-2000) map of “World Maps of the Köppen-Geiger Climate Classification” ([[link](https://datacatalog.worldbank.org/search/dataset/0042325)], Creative Commons Attribution 4.0).
